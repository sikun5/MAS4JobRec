"""Lock train-only inputs, sample and separate blind prompts from labels."""
from experiment_paths import ROOT, phase, lock_sources, source_hashes
import hashlib,itertools,json,random,sys
from pathlib import Path
RUN = phase(7)
D=RUN/'distill'
EV=RUN/'evidence'
P2=RUN.parent/'v2-phase2-20261121'
P3=RUN.parent/'v2-phase3-20261121'/'pilot'
from review_windows import PROFILE,JOB,SLOTS,SPECIALISTS
SEED=20261121
def encode(x):
    return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for buf in iter(lambda:f.read(8<<20),b''):h.update(buf)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):
    with p.open() as f:
        for line in f:yield json.loads(line)
def save(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def lines(p,x):
    p.write_text(''.join(encode(y)+'\n' for y in x))
def prepare():
    from transformers import AutoTokenizer
    from phase3_probe import TEACHER,prompt,template
    expected={name:h for h,name in (line.split(maxsplit=1) for line in (P2/'phase2.sha256').read_text().splitlines())}
    files=['data/splits.json','data/lists.jsonl','data/inputs.jsonl','labels/train.jsonl']
    for name in files:assert sha(P2/name)==expected[name]
    s=read(P2/files[0]);blocks=s['standard'];train=blocks['train']
    allq=sum(blocks.values(),[])
    for key in ('group_sha256','user_id'):
        assert len(set(s['members'][q][key] for q in allq))==len(allq)
    assert len(train)==256
    lists,inputs,labels=({r['query']:r for r in rows(P2/file) if r['query'] in train} for file in files[1:])
    assert set(train)==lists.keys()==inputs.keys()==labels.keys()
    verified=read(P3/'teacher_verified.json')
    assert TEACHER.name==verified['revision']
    for name,meta in verified['files'].items():
        p=TEACHER/name;size=p.stat().st_size
        assert size==meta['size']
        want=meta['upstream_digest']
        actual=sha(p) if len(want)==64 else hashlib.sha1(f'blob {size}\0'.encode()+p.read_bytes()).hexdigest()
        assert actual==want,name
    idx=read(TEACHER/'model.safetensors.index.json')['weight_map']
    assert len(set(idx.values()))==4 and set(idx.values())<=verified['files'].keys()
    tok=AutoTokenizer.from_pretrained(TEACHER,local_files_only=True)
    def clip(fields,limit):
        keys=sorted(fields,key=lambda k:k in ('experience','job_description'))
        txt=' '.join(f'{k}={fields[k]}' for k in keys)
        return tok.decode(tok.encode(txt,add_special_tokens=False)[:limit],skip_special_tokens=True)
    rng=random.Random(SEED)
    windows=[];requests=[];skips=[]
    for q in train:
        mapping=lists[q];ids=mapping['job_ids'];x=inputs[q]
        assert hashlib.sha256(encode(ids).encode()).hexdigest()==mapping['sha256'] and len(set(ids))==10
        assert set(x)=={'query','jobs','profile'} and set(x['profile'])==set(PROFILE)
        assert tuple(j['slot'] for j in x['jobs'])==SLOTS and all(set(j)=={'slot','fields'} and set(j['fields'])==set(JOB) for j in x['jobs'])
        truth={o['slot']:o['J'] for o in labels[q]['outcomes']}
        assert len(labels[q]['outcomes'])==10 and truth.keys()==set(SLOTS)
        sets=[v for v in itertools.combinations(SLOTS,4) if sum(truth[j] for j in v)==1]
        if not sets:skips.append(dict(query=q,reason='无一正三负组合'))
        profile=clip(x['profile'],512)
        jobs={j['slot']:clip(j['fields'],640) for j in x['jobs']}
        for n,v in enumerate(rng.sample(sets,min(2,len(sets))),1):
            ordered=list(v);rng.shuffle(ordered)
            wid=f'{q}-w{n}'
            windows.append(dict(query=q,window=wid,combination=v,order=ordered,list_sha256=mapping['sha256']))
            for role in SPECIALISTS:
                msg=prompt(tok,profile,[jobs[j] for j in ordered],role)
                size=len(tok.encode(template(tok,msg),add_special_tokens=False))
                assert size<=3840
                requests.append(dict(window=wid,query=q,role=role,messages=msg,input_tokens=size,prompt_sha256=hashlib.sha256(encode(msg).encode()).hexdigest()))
    assert len(windows)==428 and len(skips)==42 and len(requests)==1712
    lines(D/'train_windows.jsonl',windows);lines(D/'blind_requests.jsonl',requests);lines(EV/'skipped.jsonl',skips)
    kernel = P3 / 'vendor/autoawq-0.2.9/awq/modules/triton/gemm.py'
    save(D/'teacher_manifest.json',dict(seed=SEED,model=str(TEACHER),revision=verified['revision'],teacher_files=verified['files'],roles=SPECIALISTS,train=256,skipped=42,windows=428,logical_requests=1712,batch_size=2,enable_thinking=False,do_sample=False,input_cap=3840,output_cap=256,profile_cap=512,job_cap=640,source_hashes={n:expected[n] for n in files},runtime_sources={**source_hashes(7), str(kernel.relative_to(ROOT)): sha(kernel)},blind_sha256=sha(D/'blind_requests.jsonl'),windows_sha256=sha(D/'train_windows.jsonl')))
if __name__ == "__main__":
    lock_sources(7)
    prepare()
