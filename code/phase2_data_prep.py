"""Build the frozen v2 phase-2 handoff. Usage: python phase2_data_prep.py RAW_DIR NEW_RUN_DIR"""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re
import tempfile
import unicodedata

HASHES = {
    "table1_user.txt": "735ee2736d746f272dd4c0364af117c20802e5d05df5adf4aa4e406fb04262d9",
    "table2_jd.txt": "b1a05ddce12a7b14110fc48b4fbbe01489190a2e11ca768c21cac3359cba0ced",
    "table3_action.txt": "13c4759028726b8b29340ef3b1daf10267d32d8600494fdfea29e0808691316c",
}
GROUP = "cur_industry_id cur_jd_type cur_salary_id cur_degree_id start_work_date experience".split()
QUERY = "cur_industry_id cur_jd_type cur_degree_id experience desire_jd_industry_id desire_jd_type_id".split()
DOC = "jd_title jd_sub_type key min_edu_level job_description".split()
PROFILE = GROUP + "live_city_id desire_jd_city_id desire_jd_industry_id desire_jd_type_id desire_jd_salary_id".split()
JOB = DOC + "city max_salary min_salary min_years max_edu_level is_travel resume_language_required is_mangerial".split()
TOKEN = re.compile(r"[a-z]+|[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002ebef\U00030000-\U000323af]+")
SEED = 20261121


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def norm(s):
    s = " ".join(unicodedata.normalize("NFKC", s).lower().split())
    return "" if s == "-" else s


def features(row, keys):
    return {k: norm(row[k]) for k in keys}


def tokens(row, keys):
    for key in keys:
        for word in TOKEN.findall(norm(row[key])):
            if word.isascii():
                yield word
            else:
                for i in range(len(word) - 1):
                    yield word[i:i + 2]


def rows(path, *, repair=False, quarantine=None):
    with path.open(encoding="utf-8") as f:
        header = next(f).rstrip("\r\n").split("\t")
        for line_no, line in enumerate(f, 2):
            values = line.rstrip("\r\n").split("\t")
            if repair and len(values) == len(header) - 1:
                values.append("")
            if len(values) != len(header):
                if quarantine is None:
                    raise ValueError((path.name, line_no, len(values)))
                quarantine.append({"line": line_no, "fields": len(values), "jd_no": values[0]})
                continue
            yield dict(zip(header, values))


def splits_for(users, pools):
    ranked = sorted((digest(f"{SEED}|{u}"), u) for u, jobs in pools.items() if len(jobs) >= 10)
    reps = {}
    for _, u in ranked:
        group = digest(encode([norm(users[u][k]) for k in GROUP]))
        reps.setdefault(group, u)
    items = list(reps.items())
    members = {}
    blocks = {}
    start = 0
    for name, n in (("pilot", 16), ("C", 128), ("T", 256), ("train", 256)):
        block = []
        for i, (group, u) in enumerate(items[start:start+n], start):
            key = f"q{i+1:04}"
            members[key] = {"user_id": u, "group_sha256": group}
            block.append(key)
        if len(block) != n:
            raise ValueError("insufficient representatives")
        blocks[name] = block
        start += n
    small = dict(blocks, C=blocks["C"][:64], T=blocks["T"][:128])
    return {"seed": SEED, "members": members, "standard": blocks, "reduced": small}, len(items)


def rank_lists(job_file, users, members, pools, valid_count):
    # Only DF of queried terms is needed. Every valid JD contributes to DF and avgdl.
    wanted = defaultdict(list)
    queries = {}
    for q, member in members.items():
        u = member["user_id"]
        queries[q] = set(tokens(users[u], QUERY))
        for job in pools[u]:
            wanted[job].append(q)
    vocab = set.union(*queries.values())
    df = Counter()
    docs = {}
    lengths = {}
    frequencies = {}
    total_length = count = 0
    for job in rows(job_file, quarantine=[]):
        j = job["jd_no"]
        tf = Counter(tokens(job, DOC))
        length = sum(tf.values())
        total_length += length
        count += 1
        df.update(vocab.intersection(tf))
        if j in wanted:
            docs[j] = features(job, JOB)
            lengths[j] = length
            frequencies[j] = {term: tf[term] for term in set.union(*(queries[q] for q in wanted[j])).intersection(tf)}
    assert count == valid_count
    avgdl = total_length / count
    idf = {t: math.log1p((count - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    lists = {}
    for q, m in members.items():
        scores = {}
        for j in pools[m["user_id"]]:
            factor = 1.2 * (1 - .75 + .75 * lengths[j] / avgdl)
            tf = frequencies[j]
            scores[j] = math.fsum(idf[t] * tf[t] * 2.2 / (tf[t] + factor)
                                  for t in sorted(queries[q].intersection(tf)))
        lists[q] = sorted(scores, key=lambda j: (-scores[j], j))[:10]
    return lists, docs, avgdl


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def jsonlines(path, records):
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(encode(row) + "\n")


def build(raw, root):
    data = root / "data"
    labels = root / "labels"
    data.mkdir()
    labels.mkdir(mode=0o700)
    assert {name: checksum(raw/name) for name in HASHES} == HASHES
    user_file, job_file, action_file = [raw / name for name in HASHES]
    users = {r["user_id"]: r for r in rows(user_file, repair=True)}
    with user_file.open("rb") as f:
        next(f)
        user_shapes = Counter(row.rstrip(b"\r\n").count(b"\t")+1 for row in f)
    bad = []
    valid = {r["jd_no"] for r in rows(job_file, quarantine=bad)}
    merged = defaultdict(dict)
    joined = actions = 0
    for r in rows(action_file):
        actions += 1
        u, j = r["user_id"], r["jd_no"]
        if u not in users or j not in valid:
            continue
        joined += 1
        state = int(r["browsed"] + r["delivered"] + r["satisfied"], 2)
        merged[u][j] = merged[u].get(j, 0) | state
    states = Counter(format(s, "03b") for jobs in merged.values() for s in jobs.values())
    pools = {u: sorted(j for j, s in jobs.items() if s & 6) for u, jobs in merged.items()}
    touched = sum(map(len, pools.values()))
    eligible = sum(len(p) >= 10 for p in pools.values())
    assert (len(users), user_shapes[12], len(valid), len(bad), actions, joined, touched, eligible) == (
        4500, 35, 269533, 1, 700938, 648171, 129756, 3392)
    assert states == {"000":343276,"100":68463,"010":1,"110":32698,"111":28594}
    splits, groups = splits_for(users, pools)
    members = splits["members"]
    ordered, docs, avgdl = rank_lists(job_file, users, members, pools, len(valid))
    save(data/"splits.json", splits)
    jsonlines(data/"lists.jsonl", ({"query":q, "job_ids":js, "sha256":digest(encode(js))}
                                    for q, js in ordered.items()))
    jsonlines(data/"inputs.jsonl", ({"query":q, "profile":features(users[members[q]["user_id"]], PROFILE),
        "jobs":[{"slot":f"P{i:02}", "fields":docs[j]} for i,j in enumerate(js,1)]}
        for q,js in ordered.items()))
    for split, keys in splits["standard"].items():
        block = []
        for q in keys:
            outcomes = []
            for i,j in enumerate(ordered[q],1):
                state = merged[members[q]["user_id"]][j]
                d = int(bool(state & 2))
                s = state & 1
                outcomes.append({"slot":f"P{i:02}", "D":d, "S":s if d else None,
                                 "J":d*s, "delivered_mask":bool(d)})
            block.append({"query":q, "outcomes":outcomes})
        path = labels/f"{split}.jsonl"
        jsonlines(path, block)
        path.chmod(0o600)
    overlaps = {}
    for tier in ("standard","reduced"):
        jobs = {b: {j for q in qs for j in ordered[q]} for b,qs in splits[tier].items()}
        overlaps[tier] = {f"{a}/{b}":len(jobs[a] & jobs[b]) for a,b in combinations(jobs,2)}
    save(data/"audit.json", {"source_sha256":HASHES,"users":len(users),"user_row_shapes":user_shapes,
        "jd_raw":len(valid)+len(bad),"jd_valid":len(valid),"jd_quarantine":bad,"actions":actions,
        "joined_rows":joined,"unmatched_rows":actions-joined,"merged_pairs":sum(states.values()),
        "merged_states":states,"touched_pairs":touched,"eligible_users":eligible,"resume_groups":groups,
        "sampled_users":len(members),"bm25_mean_doc_tokens":avgdl,"job_overlap_counts":overlaps})
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    (root/"phase2.sha256").write_text("".join(f"{checksum(p)}  {p.relative_to(root)}\n" for p in paths))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("choose a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp)/"stage"
        stage.mkdir(mode=0o700)
        build(args.raw, stage)
        stage.rename(args.output)
