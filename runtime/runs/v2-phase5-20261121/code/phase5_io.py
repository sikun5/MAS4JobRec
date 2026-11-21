"""Shared paths and provenance for this server-side Phase 5 run."""
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
ROOT = RUN.parents[2]
P2 = RUN.parent / "v2-phase2-20261121"
P4 = RUN.parent / "v2-phase4-20261121"
BASE = RUN / "base"
ANALYSIS = RUN / "analysis"
EVIDENCE = RUN / "evidence"
PROTOCOL = ROOT / "global_guide/v2_frozen/phase5_trust_and_ranking.md"
OLD_PROTOCOL_HASH = "36c5908adad410df16cde00be2e991940d55dfeb968ed09bead3cad285bd851e"
FORMAL = ["candidate_lists.jsonl", "calibration.json", "oof.jsonl", "rankings.jsonl", "fallbacks.csv"]
ROLES = ("skill", "trajectory", "preference", "employer")
HEADS = tuple(r+".pJ" for r in ROLES) + ("preference.pC", "employer.pE")
METHODS = ("B1", "B4", "T0", "T1", "W1", "F") + tuple("B4-minus-"+r for r in ROLES)
SEED = 20261121


def encode(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def write_rows(path, values):
    temp = path.with_suffix(path.suffix+".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for v in values:
            f.write(encode(v)+"\n")
    temp.replace(path)


def table(path, values):
    temp = path.with_suffix(path.suffix+".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(values)
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""):
            h.update(block)
    return h.hexdigest()


def hashes(path):
    out = {}
    for line in path.read_text().splitlines():
        value, name = line.split(maxsplit=1)
        out[name.strip()] = value
    return out


def fingerprint(names=FORMAL):
    return {"base/"+name:sha(BASE/name) for name in names}


def save_hashes(path, values):
    path.write_text("".join(f"{h}  {p}\n" for p,h in sorted(values.items())))


def log(message):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with (RUN/"work_log.md").open("a", encoding="utf-8") as f:
        f.write(f"\n- {now}　{message}\n")
    print(message, flush=True)


class DataAccess:
    """Restrict and record data file opens in fitting, bootstrap and independent replay."""
    def __init__(self, stage):
        self.stage = stage
        self.counts = Counter()
        self.denied = []
        self.allowed = set()
        if stage in ("formal", "verify"):
            self.allowed = {P2/f for f in ("phase2.sha256","data/splits.json","data/lists.jsonl","labels/C.jsonl")}
            self.allowed |= {P4/"base/reviews.jsonl", P4/"phase4.sha256", P4/"acceptance.json"}
        self.roots = (str(P2)+"/", str(P4)+"/", str(ROOT/"data")+"/")
        sys.addaudithook(self._audit)

    def _audit(self, event, args):
        if event!="open" or not isinstance(args[0], (str,bytes,os.PathLike)):
            return
        name = os.path.abspath(os.fsdecode(args[0]))
        if not name.startswith(self.roots):
            return
        path = Path(name)
        rel = str(path.relative_to(ROOT))
        if path not in self.allowed or args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
            self.denied.append(rel)
            raise PermissionError("Phase 5 data access denied: "+rel)
        self.counts[rel]+=1

    def save(self):
        write(EVIDENCE/(self.stage+"_data_access.json"),{
            "stage":self.stage, "opened_upstream_data_files":dict(sorted(self.counts.items())),
            "denied":self.denied,"test_label_opens":sum(n for p,n in self.counts.items() if p.endswith("/labels/T.jsonl"))})


def load_sources():
    h2 = hashes(P2/"phase2.sha256")
    h4 = hashes(P4/"phase4.sha256")
    checked = {}
    for root, manifest, names in [(P2,h2,["data/splits.json","data/lists.jsonl","labels/C.jsonl"]),
                                  (P4,h4,["base/reviews.jsonl","acceptance.json"])]:
        for name in names:
            actual = sha(root/name)
            assert actual==manifest[name], name
            checked[str((root/name).relative_to(ROOT))] = actual
    split = read(P2/"data/splits.json")
    c,t = (split["standard"][k] for k in ("C","T"))
    assert len(set(c))==128 and len(set(t))==256 and set(c).isdisjoint(t)
    hashed = [hashlib.sha256(f"{SEED}|{split['members'][q]['user_id']}".encode()).hexdigest() for q in c]
    assert hashed==sorted(hashed)
    wanted = set(c+t)
    listing = {r["query"]:r for r in rows(P2/"data/lists.jsonl") if r["query"] in wanted}
    assert set(listing)==wanted
    for r in listing.values():
        assert len(r["job_ids"])==len(set(r["job_ids"]))==10
        assert hashlib.sha256(encode(r["job_ids"]).encode()).hexdigest()==r["sha256"]
    truth = {r["query"]:r["outcomes"] for r in rows(P2/"labels/C.jsonl")}
    assert set(truth)==set(c)
    reviews = {}
    for row in rows(P4/"base/reviews.jsonl"):
        q,role = row["query"],row["role"]
        assert q in wanted and (q,role) not in reviews and row["block"]==("C" if q in truth else "T")
        assert row["list_sha256"]==listing[q]["sha256"]
        if row["status"]=="valid":
            ids = listing[q]["job_ids"]
            assert len(row["order_job_ids"])==10 and set(row["order_job_ids"])==set(ids)
            assert len(row["predictions"])==10
            assert all(p["slot"]==f"P{i:02}" and p["job_id"]==j for i,(p,j) in enumerate(zip(row["predictions"],ids),1))
        else:
            assert row["status"]=="missing"
        reviews[q,role]=row
    assert set(reviews)=={(q,r) for q in c+t for r in ROLES+(("general",) if q not in truth else ())}
    return c,t,listing,truth,reviews,checked
