"""Audit the completed pilot against the frozen protocol. Run from project root."""
import csv
import json
import math
from collections import Counter
from experiment_paths import phase

D=phase(3) / "pilot"
S=phase(2)
roles={"skill","trajectory","preference","employer"}
def rows(name):
    return [json.loads(l) for l in (D/(name+".jsonl")).read_text().splitlines() if l]

def check():
    pilot=set(json.loads((S/"data/splits.json").read_text())["standard"]["pilot"])
    truth={x["query"]:[j["J"] for j in x["outcomes"]] for l in (S/"labels/pilot.jsonl").read_text().splitlines() if (x:=json.loads(l))}
    assert set(truth)==pilot
    meta=json.loads((D/"hf.json").read_text())
    verified=json.loads((D/"teacher_verified.json").read_text())
    assert verified["revision"]==meta["sha"]=="0499c3ac83fdef8810b907a23894ba91e95eddd8"
    shards=[]
    for f in meta["siblings"]:
        n=f["rfilename"]; v=verified["files"][n]
        digest=f["lfs"]["sha256"] if "lfs" in f else f["blobId"]
        assert v["upstream_digest"]==digest and v["size"]==f["size"]
        if n.endswith(".safetensors"):shards.append(n)
    assert len(shards)==4
    training_samples={}
    for mode in ("base","teacher","full","lora"):
        r=rows(mode)
        assert r[0]["name"]=="NVIDIA H100 80GB HBM3" and r[-1]["event"]=="completed",mode
        inf=[x for x in r if x["event"]=="inference"]
        tested=roles|{"general"} if mode=="base" else roles
        for role in tested:
            for band in ("typical","long"):
                group=[x for x in inf if x["band"]==band and x["role"]==role]
                assert len(group)>=2
                assert all(x["seconds"]>0 and set(q for q,_ in x["query_start"])<=pilot and
                           max(x["inputs"])<=3840 and x["peak_allocated"]>0 and len(x["valid"])==x["windows"] for x in group)
        if mode in ("full","lora"):
            steps=[x for x in r if x["event"]=="train_step"]
            training_samples[mode]=[(e["query"],tuple(e["window"]),e["role"]) for e in steps]
            assert len(steps)>=4 and all(math.isfinite(x["loss"]) and x["grad_norm"]>0 and x["supervised"]>0 for x in steps)
            assert any(x["delta"]>0 for x in steps)
            assert all((x["trainable"]==x["parameters"])==(mode=="full") for x in steps)
            assert any(x["event"]=="save" for x in r) and any(x["event"]=="activate_student" for x in r)
        sr=rows("base_stress") if mode=="base" else r
        assert any(x["event"]=="stress" and x["inputs"]==3840 and x["generated"]==256 for x in sr)
    retained=json.loads((D/"teacher_demos.json").read_text())
    extra=json.loads((D/"teacher_supplement_demos.json").read_text()) if (D/"teacher_supplement_demos.json").exists() else []
    if extra:
        def teacher_keys(name):
            return {(query,tuple(window),e["role"]) for e in rows(name) if e["event"]=="inference" for query,window in e["query_start"]}
        assert teacher_keys("teacher").isdisjoint(teacher_keys("teacher_supplement"))
    demo_keys={(x["query"],tuple(x["start"]),x["role"]) for x in retained+extra}
    assert training_samples["full"]==training_samples["lora"] and set(training_samples["full"])<=demo_keys
    for x in retained+extra:
        top=json.loads(x["answer"].strip().removeprefix("```json").removesuffix("```").strip())["rank"][0]
        y=[truth[x["query"]][i] for i in x["start"]]
        assert sum(y)==1 and y[top-1]==1 and x["role"] in roles
    with (D/"runtime_estimates.csv").open() as f:
        estimates=list(csv.DictReader(f))
    assert len(estimates)==3
    for z in estimates:
        for k in ["wall","gpu"]:
            lo,center,hi=[float(z[k+"_"+j+"_h"]) for j in ("low","center","high")]
            assert 0<lo<=center<=hi
    return dict(verified_shards=len(shards),pilot_user_pool=len(pilot),retained_demos=len(retained),
                supplemental_retained=len(extra),matched_training_sequences=True,
                retained_per_role=dict(Counter(x["role"] for x in retained)),three_routes_verified=True)

if __name__=="__main__":
    result=check()
    print(json.dumps(result))
    (D/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
