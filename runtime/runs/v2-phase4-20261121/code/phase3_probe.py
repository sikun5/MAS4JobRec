"""Single-H100 Phase 3 timing probe. Run each mode in a separate process."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PILOT = Path("runtime/runs/v2-phase3-20261121/pilot")
SOURCE = Path("runtime/runs/v2-phase2-20261121/data")
BASE = Path("model/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218")
TEACHER = Path("model/Qwen3-32B-AWQ/0499c3ac83fdef8810b907a23894ba91e95eddd8")
ROLES = {"skill":"技能证据", "trajectory":"职业经历及岗位跨度", "preference":"求职意向", "employer":"招聘要求", "general":"综合考虑技能、经历、意向与招聘要求"}


def dump(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def emit(event, **fields):
    torch.cuda.synchronize()
    row = dict(event=event, epoch=time.time(), peak_allocated=torch.cuda.max_memory_allocated(),
               peak_reserved=torch.cuda.max_memory_reserved(), free=torch.cuda.mem_get_info()[0], **fields)
    with (PILOT / (MODE+".jsonl")).open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False)+"\n")
    print(event,fields.get("role",""),fields.get("band",""),fields.get("seconds",""),flush=True)


def prompt(tok, profile, jobs, role):
    extra = {"preference": "，另给四岗投递概率pC", "employer":"，另给四岗假设已投递的认可概率pE"}.get(role,"")
    fmt = '{"rank":[2,1,4,3],"pJ":[0.3,0.6,0.2,0.25]' + {"preference":',"pC":[0.6,0.8,0.4,0.5]',"employer":',"pE":[0.5,0.75,0.5,0.5]'}.get(role,"") + '}'
    return [{"role":"system","content":"你是岗位评审。重点看"+ROLES[role]+"。按投递且获雇主认可的可能性降序排列四岗，估计每岗联合概率pJ"+extra+"。所有概率数组按岗位编号1至4排列，范围0到1。只输出一个JSON，示例结构："+fmt},
            {"role":"user","content":"求职资料："+profile+"\n"+"\n".join(f"岗位{i+1}：{v}" for i,v in enumerate(jobs))}]


def template(tok,messages):
    return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)


def windows(tok,teacher=False,supplement=False):
    ids=set(json.loads((SOURCE/"splits.json").read_text())["standard"]["pilot"])
    candidates=[]
    if teacher:
        import itertools
        import random
        rng=random.Random(20261121)
        truths={v["query"]:[j["J"] for j in v["outcomes"]] for line in (SOURCE.parent/"labels/pilot.jsonl").read_text().splitlines() if (v:=json.loads(line))}
    def clip(fields,cap):
        keys=sorted(fields,key=lambda k:k in ("experience","job_description"))
        txt=" ".join(f"{k}={fields[k]}" for k in keys)
        return tok.decode(tok.encode(txt,add_special_tokens=False)[:cap],skip_special_tokens=True)
    with (SOURCE/"inputs.jsonl").open() as f:
        for line in f:
            x=json.loads(line)
            if x["query"] not in ids:continue
            profile=clip(x["profile"],512)
            choices=[(i,tuple(range(i,i+4))) for i in [6,4,2,0]]
            if teacher:
                sets=[v for v in itertools.combinations(range(10),4) if sum(truths[x["query"]][j] for j in v)==1]
                choices=[]
                for selected in rng.sample(sets,min(2,len(sets))):
                    order=list(selected);rng.shuffle(order)
                    choices.append((tuple(order),order))
            for start,order in choices:
                jobs=[clip(x["jobs"][j]["fields"],640) for j in order]
                length=len(tok.encode(template(tok,prompt(tok,profile,jobs,"skill")),add_special_tokens=False))
                candidates.append((length,x["query"],start,profile,jobs))
    candidates.sort(key=lambda a:a[:3])
    mid=len(candidates)//2
    if supplement:
        # Additional pilot windows around the median, disjoint from both initial bands.
        # Keep the original two sampled combinations per person.
        quarter=len(candidates)//4
        return {"typical":candidates[quarter:mid-2]+candidates[mid+2:len(candidates)-quarter]}
    return {"typical":candidates[mid-2:mid+2],"long":candidates[-4:]}


def parse(text,role):
    try:
        obj=json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
        assert sorted(obj["rank"])==[1,2,3,4]
        for key in ["pJ"]+({"preference":["pC"],"employer":["pE"]}.get(role,[])):
            ps=obj[key];assert len(ps)==4 and all(isinstance(p,(int,float)) and 0<=p<=1 for p in ps)
        return True
    except (TypeError,ValueError,KeyError,AssertionError,AttributeError):
        return False


def generate(model,tok,msgs):
    p=[template(tok,m) for m in msgs]
    x=tok(p,add_special_tokens=False,padding=True,return_tensors="pt").to("cuda")
    assert x.input_ids.shape[1]<=3840
    with torch.inference_mode():
        y=model.generate(**x,max_new_tokens=256,do_sample=False,pad_token_id=tok.pad_token_id)
    out=y[:,x.input_ids.shape[1]:]
    texts=tok.batch_decode(out,skip_special_tokens=True)
    return texts,x.attention_mask.sum(dim=1).tolist(),[(v!=tok.pad_token_id).sum().item() for v in out]


def inference(model,tok,roles,write_demos=False):
    model.eval()
    supplement=MODE=="teacher_supplement"
    samples=windows(tok,teacher=write_demos,supplement=supplement);demos=[]
    if write_demos:
        truths={v["query"]:[j["J"] for j in v["outcomes"]] for line in (SOURCE.parent/"labels/pilot.jsonl").read_text().splitlines() if (v:=json.loads(line))}
    for band,chunk in samples.items():
        for role in roles:
            for rep in range((len(chunk)+1)//2):
                batch=chunk[rep*2:rep*2+2]
                msgs=[prompt(tok,c[3],c[4],role) for c in batch]
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                texts,ins,outs=generate(model,tok,msgs)
                retries=0
                for i,t in enumerate(texts):
                    if not parse(t,role):
                        retries+=1
                        tx,_,ox=generate(model,tok,[msgs[i]]);texts[i]=tx[0];outs[i]+=ox[0]
                torch.cuda.synchronize();elapsed=time.monotonic()-start
                valid=[parse(t,role) for t in texts]
                emit("inference",band=band,role=role,seconds=elapsed,rep=rep,windows=len(batch),inputs=ins,outputs=outs,retries=retries,valid=valid,
                     query_start=[list(c[1:3]) for c in batch],responses=texts)
                if write_demos:
                    for msg,t,v,c in zip(msgs,texts,valid,batch):
                        if v and truths[c[1]][c[2][json.loads(t.strip().removeprefix("```json").removesuffix("```").strip())["rank"][0]-1]]:demos.append(dict(messages=msg,answer=t,role=role,band=band,query=c[1],start=c[2]))
    if write_demos:dump(PILOT/(MODE+"_demos.json"),demos)
    if supplement:return
    from phase3_stress import stress
    c=samples["long"][-1]
    stress(model,tok,prompt(tok,c[3],c[4],"employer"),emit)


def main():
    torch.manual_seed(20261121)
    PILOT.mkdir(exist_ok=True,parents=True)
    dev=torch.cuda.get_device_properties(0)
    emit("environment",name=dev.name,total=dev.total_memory,torch=torch.__version__,cuda=torch.version.cuda,mode=MODE)
    is_teacher=MODE in ("teacher","teacher_supplement")
    src=TEACHER if is_teacher else BASE
    t=time.monotonic()
    tok=AutoTokenizer.from_pretrained(src,local_files_only=True,padding_side="left")
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    if is_teacher:
        from phase3_awq import load_awq
        model=load_awq(src)
    else:
        model=AutoModelForCausalLM.from_pretrained(src,dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa",local_files_only=True)
    torch.cuda.synchronize()
    emit("load",seconds=time.monotonic()-t,dtype=str(next(model.parameters()).dtype))
    if MODE=="base" or is_teacher:
        t=time.monotonic()
        one=windows(tok)["typical"][0];generate(model,tok,[prompt(tok,one[3],one[4],"skill")])
        torch.cuda.synchronize();emit("warmup",seconds=time.monotonic()-t)
        inference(model,tok,list(ROLES) if MODE=="base" else list(ROLES)[:4],is_teacher)
    elif MODE=="base_stress":
        from phase3_stress import stress
        c=windows(tok)["long"][-1]
        stress(model,tok,prompt(tok,c[3],c[4],"employer"),emit)
    else:
        from phase3_train import train
        train(model,tok,MODE,PILOT,emit,template,inference)
    emit("completed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["base","teacher","teacher_supplement","full","lora","base_stress"])
    MODE=ap.parse_args().mode
    try:main()
    except Exception as exc:
        with (PILOT/(MODE+".jsonl")).open("a") as f:f.write(json.dumps(dict(event="error",error=repr(exc),epoch=time.time(),peak_allocated=torch.cuda.max_memory_allocated()))+"\n")
        raise
