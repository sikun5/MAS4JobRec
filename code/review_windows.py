"""Shared parsing, sliding-window ballots and checkpointed inference for both 8B models."""
from experiment_paths import ROOT, phase
import json
import math
import os
import sqlite3
import time
from collections import defaultdict

from phase2_data_prep import PROFILE, JOB, digest, encode, checksum

DATA = phase(2) / "data"
MODEL = ROOT / "model/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218"
SPECIALISTS = ("skill", "trajectory", "preference", "employer")
STARTS = (6, 4, 2, 0) * 2
SLOTS = tuple(f"P{i:02}" for i in range(1, 11))
BATCH = 2
DATA_HASHES = {"inputs.jsonl":"d57df513b5bb752ef15f5f5bf14349e6e1e90a18116b1e84e7d0fffae348d064",
               "lists.jsonl":"91d481e816bb3a0fe903761880837c72b9644600f3abbcf9a1788432596512e8",
               "splits.json":"0cab3b846ca7a051c5c7a4d4a3ba744ae9589811222e68afb74f6b14052243df"}


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    tmp.replace(path)


def journal(run, message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
    with (run/"work_log.md").open("a") as f:
        f.write(f"\n- {stamp}　{message}\n")
    print(stamp, message, flush=True)


def parse(raw, role):
    text = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("JSON object required")
        rank = obj["rank"]
        if not isinstance(rank, list) or any(type(i) is not int for i in rank) or sorted(rank)!=[1,2,3,4]:
            raise ValueError("rank must permute 1..4")
        keys = ["pJ"] + {"preference":["pC"], "employer":["pE"]}.get(role, [])
        for key in keys:
            p = obj[key]
            if not isinstance(p, list) or len(p)!=4 or any(type(x) not in (int,float) or not 0<=x<=1 for x in p):
                raise ValueError("invalid "+key)
        return {"rank":rank, **{k:obj[k] for k in keys}}, None
    except (KeyError, TypeError, ValueError) as exc:
        return None, str(exc)


def load_data():
    for name, expected in DATA_HASHES.items():
        assert checksum(DATA/name)==expected, name
    split = json.loads((DATA/"splits.json").read_text())["standard"]
    ids = {q for block in ["C","T"] for q in split[block]}
    assert len(split["C"])==128 and len(split["T"])==256 and len(ids)==384
    def listed(name):
        return {x["query"]:x for line in (DATA/name).open() if (x:=json.loads(line))["query"] in ids}
    inputs, lists = listed("inputs.jsonl"), listed("lists.jsonl")
    assert set(inputs)==ids==set(lists)
    for q in ids:
        x, mapping = inputs[q], lists[q]
        assert set(x)=={"query","jobs","profile"} and set(x["profile"])==set(PROFILE)
        assert tuple(j["slot"] for j in x["jobs"])==SLOTS
        assert all(set(j)=={"slot","fields"} and set(j["fields"])==set(JOB) for j in x["jobs"])
        assert len(set(mapping["job_ids"]))==10 and digest(encode(mapping["job_ids"]))==mapping["sha256"]
    return split, inputs, lists


class Review:
    def __init__(self, block, q, role, mapping, run):
        self.run = run
        self.block, self.q, self.role = block, q, role
        self.slots = list(SLOTS)
        self.jobs = dict(zip(SLOTS,mapping["job_ids"]))
        self.list_hash = mapping["sha256"]
        self.step = 0
        self.obs = defaultdict(lambda: defaultdict(list))
        self.failed = False
        self.windows = []

    def identity(self):
        start = STARTS[self.step]
        slots = self.slots[start:start+4]
        fields = dict(run=self.run.name, model_revision=MODEL.name, block=self.block, query=self.q, role=self.role,
                      list_sha256=self.list_hash, sweep=1+self.step//4, start=start+1, input_slots=slots,
                      input_job_ids=[self.jobs[s] for s in slots])
        return {"cache_key":digest(encode(fields)), **fields}

    def finish(self, attempts):
        row = self.identity()
        last = attempts[-1]
        obj = last.get("parsed")
        updated = []
        if obj is None:
            assert len(attempts)==2
            self.failed=True
        else:
            slots = row["input_slots"]
            updated = [slots[i-1] for i in obj["rank"]]
            for key, values in obj.items():
                if key!="rank":
                    for slot, p in zip(slots,values):
                        p=min(.9999,max(.0001,p))
                        self.obs[slot][key].append(math.log(p/(1-p)))
            pos=STARTS[self.step]
            self.slots[pos:pos+4]=updated
        self.windows.append(dict(**row, attempts=attempts, status="missing" if self.failed else "valid",
                                 output_slots=updated, output_job_ids=[self.jobs[s] for s in updated]))
        self.step+=1

    def result(self):
        assert self.failed or self.step==8
        predictions=[]
        if not self.failed:
            for slot in SLOTS:
                heads=self.obs[slot]
                assert "pJ" in heads
                probs={k:1/(1+math.exp(-math.fsum(vals)/len(vals))) for k,vals in heads.items()}
                predictions.append(dict(slot=slot, job_id=self.jobs[slot], observations=len(heads["pJ"]), **probs))
        return dict(block=self.block, query=self.q, role=self.role, list_sha256=self.list_hash,
                    status="missing" if self.failed else "valid", order_slots=[] if self.failed else self.slots,
                    order_job_ids=[] if self.failed else [self.jobs[s] for s in self.slots],
                    predictions=predictions, finished_windows=self.step,
                    skipped=[dict(sweep=1+i//4,start=STARTS[i]+1) for i in range(self.step,8)])


def connect(path):
    db=sqlite3.connect(path)
    db.execute("pragma synchronous=FULL")
    db.execute("""create table if not exists attempts (
        key text, n integer, state text, payload text, primary key(key,n))""")
    return db


def attempt_rows(db, key):
    return [json.loads(s[0]) for s in db.execute("select payload from attempts where key=? order by n",(key,))]


def complete_attempts(db, model, tok, requests, number):
    from phase3_probe import generate
    import torch
    # Persist dispatch before GPU execution so a crash can be distinguished from a cache hit.
    if not requests:return
    stamp=time.time()
    batch_id=f"{os.getpid()}-{time.time_ns()}"
    records=[]
    for review, messages in requests:
        key=review.identity()["cache_key"]
        a=dict(number=number, batch_id=batch_id, batch_size=len(requests), started_epoch=stamp,
               prompt_sha256=digest(encode(messages)), raw=None, parsed=None, error="dispatch_pending")
        records.append((key,a))
    with db:
        db.executemany("insert into attempts values (?,?,?,?)",
                       [(key,number,"pending",encode(a)) for key,a in records])
    torch.cuda.synchronize()
    started=time.monotonic()
    raw,ins,outs=generate(model,tok,[m for _,m in requests])
    torch.cuda.synchronize()
    elapsed=time.monotonic()-started
    assert len(raw)==len(requests) and all(x<=3840 for x in ins) and all(x<=256 for x in outs)
    with db:
        for (key,a),(r,_),response,it,ot in zip(records,requests,raw,ins,outs):
            obj, error=parse(response,r.role)
            a.update(raw=response,parsed=obj,error=error,input_tokens=it,output_tokens=ot,
                     batch_seconds=elapsed,allocated_seconds=elapsed/len(requests),ended_epoch=time.time())
            db.execute("update attempts set state=?,payload=? where key=? and n=?",
                       ("done",encode(a),key,number))


def play_block(reviews, documents, tok, model, db):
    from phase3_probe import prompt
    for step in range(8):
        active=[r for r in reviews if not r.failed]
        for offset in range(0,len(active),BATCH):
            group=active[offset:offset+BATCH]
            requests=[]
            for r in group:
                assert r.step==step
                profile,jobs=documents[r.q]
                messages=prompt(tok,profile,[jobs[s] for s in r.identity()["input_slots"]],r.role)
                requests.append((r,messages))
            for n in (1,2):
                due=[]
                for r,messages in requests:
                    key=r.identity()["cache_key"]
                    logged=attempt_rows(db,key)
                    assert all(a["prompt_sha256"]==digest(encode(messages)) for a in logged)
                    if len(logged)==n-1 and not any(a["parsed"] for a in logged):
                        due.append((r,messages))
                if n==1:
                    complete_attempts(db,model,tok,due,n)
                else:
                    for retry in due:complete_attempts(db,model,tok,[retry],n)
            for r,_ in requests:r.finish(attempt_rows(db,r.identity()["cache_key"]))
        count=db.execute("select count(*) from attempts").fetchone()[0]
        journal(reviews[0].run, f'{reviews[0].block} / {reviews[0].role} 完成第 {step+1}/8 轮写回，累计记录 {count} 次请求。')
