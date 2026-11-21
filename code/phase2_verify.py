"""Verify a phase-2 handoff against the raw TSVs, without scoring the test partition."""
import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re
import stat
import unicodedata


def sha(data):
    return hashlib.sha256(data).hexdigest()


def serialized(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


def clean(s):
    value = " ".join(unicodedata.normalize("NFKC", s).lower().split())
    return "" if value == "-" else value


def tsv(path):
    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(raw, run):
    data = run/"data"
    audit = json.loads((data/"audit.json").read_text())
    splits = json.loads((data/"splits.json").read_text())
    members = splits["members"]
    candidates = load_rows(data/"inputs.jsonl")
    lists = load_rows(data/"lists.jsonl")
    assert len(members) == len(candidates) == len(lists) == 656
    assert {r["query"] for r in lists} == {r["query"] for r in candidates} == set(members)
    ordered = {}
    for row in lists:
        q, js = row["query"], row["job_ids"]
        assert len(js) == len(set(js)) == 10 and row["sha256"] == sha(serialized(js))
        ordered[q] = js
    source_users = tsv(raw/"table1_user.txt")
    hdr = next(source_users)
    users = {row[0]:dict(zip(hdr, row if len(row)==len(hdr) else row+[""])) for row in source_users}
    jobs_iter = tsv(raw/"table2_jd.txt")
    hdr = next(jobs_iter)
    chosen = set().union(*map(set,ordered.values()))
    valid_jobs = set()
    chosen_fields = {}
    for row in jobs_iter:
        if len(row)==len(hdr):
            valid_jobs.add(row[0])
            if row[0] in chosen:
                chosen_fields[row[0]]=dict(zip(hdr,row))
    merged = defaultdict(dict)
    actions = joined = 0
    acts = tsv(raw/"table3_action.txt")
    next(acts)
    for u,j,b,d,s in acts:
        actions+=1
        if j in valid_jobs and u in users:
            joined+=1
            old=merged[u].get(j,(0,0,0))
            merged[u][j]=tuple(map(max,old,map(int,(b,d,s))))
    states=Counter(v for a in merged.values() for v in a.values())
    pools={u:{j for j,(b,d,s) in jobs.items() if b or d} for u,jobs in merged.items()}
    eligible=sorted((u for u in pools if len(pools[u])>=10),
                    key=lambda u:(sha(f"20261121|{u}".encode()),u))
    assert (len(users),len(valid_jobs),actions,joined,sum(states.values()),sum(map(len,pools.values())),len(eligible))==(4500,269533,700938,648171,473032,129756,3392)
    assert states[0,1,0]==1
    group_keys="cur_industry_id cur_jd_type cur_salary_id cur_degree_id start_work_date experience".split()
    reps={}
    for u in eligible:
        group=sha(serialized([clean(users[u][k]) for k in group_keys]))
        reps.setdefault(group,u)
    block_names=("pilot","C","T","train")
    qs=[q for b in block_names for q in splits["standard"][b]]
    assert [(members[q]["group_sha256"],members[q]["user_id"]) for q in qs]==list(reps.items())[:656]
    assert len(set(qs))==656
    for tier, sizes in (("standard",(16,128,256,256)),("reduced",(16,64,128,256))):
        blocks=splits[tier]
        for b,n in zip(block_names,sizes):
            assert len(blocks[b])==n and blocks[b]==splits["standard"][b][:n]
        for a,b in combinations(block_names,2):
            left={j for q in blocks[a] for j in ordered[q]}
            right={j for q in blocks[b] for j in ordered[q]}
            assert len(left & right)==audit["job_overlap_counts"][tier][f"{a}/{b}"]
    profile=set(group_keys+"live_city_id desire_jd_city_id desire_jd_industry_id desire_jd_type_id desire_jd_salary_id".split())
    job=set("jd_title jd_sub_type key min_edu_level job_description city max_salary min_salary min_years max_edu_level is_travel resume_language_required is_mangerial".split())
    raw_ids=set(users)|valid_jobs
    for row in candidates:
        q=row["query"]
        u=members[q]["user_id"]
        assert set(row)=={"query","profile","jobs"}
        assert row["profile"]=={k:clean(users[u][k]) for k in profile}
        assert not raw_ids.intersection(re.findall(r'[a-f0-9]{32}',json.dumps(row,ensure_ascii=False)))
        assert set(ordered[q]) <= pools[u]
        assert len(row["jobs"])==10
        for i,(item,j) in enumerate(zip(row["jobs"],ordered[q]),1):
            assert item=={"slot":f"P{i:02}","fields":{k:clean(chosen_fields[j][k]) for k in job}}
    checked=0
    for b in block_names:
        path=run/"labels"/f"{b}.jsonl"
        assert stat.S_IMODE(path.stat().st_mode)==0o600
        rows=load_rows(path)
        assert [r["query"] for r in rows]==splits["standard"][b]
        for row in rows:
            q=row["query"]
            u=members[q]["user_id"]
            assert len(row["outcomes"])==10
            for i,(out,j) in enumerate(zip(row["outcomes"],ordered[q]),1):
                _,d,s=merged[u][j]
                assert out=={"slot":f"P{i:02}","D":d,"S":s if d else None,"J":d*s,"delivered_mask":bool(d)}
                checked+=1
    assert stat.S_IMODE((run/"labels").stat().st_mode)==0o700
    manifest=(run/"phase2.sha256").read_text()
    checked_files=set()
    for line in manifest.splitlines():
        expected,rel=line.split("  ",1)
        assert sha((run/rel).read_bytes())==expected
        checked_files.add(rel)
    assert checked_files=={"data/"+n for n in ("audit.json","splits.json","lists.jsonl","inputs.jsonl")}|{f"labels/{b}.jsonl" for b in block_names}
    return {"users_checked":len(qs),"resume_groups_before_sampling":len(reps),"input_positions_checked":checked,
            "source_label_agreement":True,"interblock_user_overlap":0,"interblock_profile_overlap":0,
            "nested_tiers_verified":True,"touch_membership_verified":True,"blind_allowlist_verified":True,
            "test_labels_mode":"0600","labels_directory_mode":"0700","manifest_files":len(checked_files),
            "manifest_sha256":sha(manifest.encode())}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw",type=Path)
    parser.add_argument("run",type=Path)
    a=parser.parse_args()
    print(json.dumps(verify(a.raw,a.run),indent=2))
