"""Run the original specialists and the test-set generalist."""
import json
import os
import time

from experiment_paths import lock_sources, phase, source_hashes
from phase2_data_prep import JOB, PROFILE, checksum, encode
from review_windows import (BATCH, DATA_HASHES, MODEL, SPECIALISTS, STARTS, Review,
                            connect, journal, load_data, play_block, write_json)

RUN = phase(4)


def roles_for(block):
    return SPECIALISTS + (("general",) if block == "T" else ())


def run():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from phase3_probe import prompt, template
    import transformers
    import torch
    torch.manual_seed(20261121)
    split,inputs,lists=load_data()
    assert torch.cuda.device_count()==1 and "H100" in torch.cuda.get_device_name()
    sources = source_hashes(4)
    cfg=dict(model=str(MODEL),device=torch.cuda.get_device_name(),torch=torch.__version__,
             transformers=transformers.__version__,cuda=torch.version.cuda,dtype="bfloat16",attention="sdpa",
             seed=20261121,do_sample=False,enable_thinking=False,batch_size=BATCH,input_cap=3840,output_cap=256,
             profile_cap=512,job_cap=640,starts=[x+1 for x in STARTS],input_hashes=DATA_HASHES,
             model_manifest_sha256=checksum(phase(1) / "env/model_files.sha256"),
             source_sha256=sources, model_inputs={"profile":PROFILE,"job":JOB})
    conf=RUN/"manifest.json"
    if conf.exists():assert json.loads(conf.read_text())==cfg,"incompatible resume configuration"
    else:write_json(conf,cfg)
    db=connect(RUN / "base/attempts.sqlite")
    receipt=RUN/"resume_log.jsonl"
    old=db.execute("select count(*) from attempts").fetchone()[0]
    interrupted=list(db.execute("select key,n,payload from attempts where state='pending'"))
    with db:
        for key,n,payload in interrupted:
            a=json.loads(payload);a["error"]="interrupted_after_dispatch"
            db.execute("update attempts set state='interrupted',payload=? where key=? and n=?",(encode(a),key,n))
    with receipt.open("a") as f:f.write(encode(dict(epoch=time.time(),pid=os.getpid(),existing=old,
                                                      interrupted=[[k,n] for k,n,_ in interrupted]))+"\n")
    journal(RUN, f"开始原版评审，已核对 {len(inputs)} 份输入的名单哈希。恢复时已有 {old} 条请求记录。")
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side="left")
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    def clip(fields,cap):
        keys=sorted(fields,key=lambda k:k in ("experience","job_description"))
        source=" ".join(f"{k}={fields[k]}" for k in keys)
        return tok.decode(tok.encode(source,add_special_tokens=False)[:cap],skip_special_tokens=True)
    docs={q:(clip(x["profile"],512),{j["slot"]:clip(j["fields"],640) for j in x["jobs"]}) for q,x in inputs.items()}
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":"cuda:0"},
                                                attn_implementation="sdpa",local_files_only=True).eval()
    all_reviews=[]
    for block in ["C","T"]:
        roles=roles_for(block)
        for role in roles:
            current=[Review(block,q,role,lists[q],RUN) for q in split[block]]
            play_block(current,docs,tok,model,db)
            all_reviews.extend(current)
    paths={k:RUN/"base"/(k+".jsonl") for k in ("windows","reviews","failures")}
    streams={k:p.with_suffix(".tmp").open("w") for k,p in paths.items()}
    for r in all_reviews:
        result=r.result()
        streams["reviews"].write(encode(result)+"\n")
        for w in r.windows:
            streams["windows"].write(encode(w)+"\n")
            if len(w["attempts"])>1:
                streams["failures"].write(encode(dict(cache_key=w["cache_key"],query=r.q,block=r.block,role=r.role,
                    sweep=w["sweep"],start=w["start"],errors=[a["error"] for a in w["attempts"]],
                    status=w["status"],skipped=result["skipped"] if w["status"]=="missing" else []))+"\n")
    for k,p in paths.items():
        streams[k].close()
        p.with_suffix(".tmp").replace(p)
    db.close()
    journal(RUN, "九个推理块已写回，窗口、选票与失败明细已导出，待独立重放验收。")


if __name__ == "__main__":
    lock_sources(4)
    run()
