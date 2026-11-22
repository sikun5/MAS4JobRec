"""Resume from committed per-request attempt records. Reads only blind requests."""
from experiment_paths import lock_sources
import gc,sqlite3,time
import torch,transformers
from phase7_prepare_windows import (D, EV, P3, SEED, encode, read, rows, save, sha)
from review_windows import parse
from phase3_probe import TEACHER,generate
from phase3_awq import load_awq
def main():
    manifest=read(D/'teacher_manifest.json')
    assert sha(D/'blind_requests.jsonl')==manifest['blind_sha256']
    assert sha(P3/'vendor/autoawq-0.2.9/awq/modules/triton/gemm.py')==read(P3/'awq_source.json')['kernel_sha256']
    db=sqlite3.connect(D/'attempts.sqlite')
    db.execute('create table if not exists attempts(window text,role text,n integer,state text,payload text,primary key(window,role,n))')
    db.execute("update attempts set state='interrupted' where state='pending'");db.commit()
    assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name()
    tok=transformers.AutoTokenizer.from_pretrained(TEACHER,local_files_only=True,padding_side='left')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    torch.manual_seed(SEED)
    t=time.monotonic();model=load_awq(TEACHER);torch.cuda.synchronize()
    save(EV/'load.json',dict(torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),seconds=time.monotonic()-t))
    rec=list(rows(D/'blind_requests.jsonl'))
    todo=[]
    for r in sorted(rec,key=lambda x:(x['role'],x['window'])):
        prior=db.execute('select state from attempts where window=? and role=? order by n',(r['window'],r['role'])).fetchall()
        if ('valid',) not in prior and len(prior)<2:todo.append((r,len(prior)+1))
    def attempt(batch):
        with db:
            for r,n in batch:
                data=dict(window=r['window'],query=r['query'],role=r['role'],attempt=n,prompt_sha256=r['prompt_sha256'],input_tokens=r['input_tokens'],output_tokens=None,started=time.time(),raw=None,parsed=None,error='interrupted_after_dispatch')
                db.execute('insert into attempts values(?,?,?,?,?)',(r['window'],r['role'],n,'pending',encode(data)))
        start=time.monotonic()
        try:
            texts,it,ot=generate(model,tok,[r['messages'] for r,n in batch])
            torch.cuda.synchronize();failure=None
        except Exception as exc:
            failure=type(exc).__name__+': '+str(exc)
            texts=[None]*len(batch);it=[r['input_tokens'] for r,n in batch];ot=[None]*len(batch)
            gc.collect();torch.cuda.empty_cache()
        seconds=time.monotonic()-start
        retries=[]
        with db:
            for (r,n),raw,i,o in zip(batch,texts,it,ot):
                assert i==r['input_tokens']
                obj,error=parse(raw,r['role']) if raw is not None else (None,failure)
                data=dict(window=r['window'],query=r['query'],role=r['role'],attempt=n,prompt_sha256=r['prompt_sha256'],input_tokens=i,output_tokens=o,finished=time.time(),batch_seconds=seconds,batch_size=len(batch),raw=raw,parsed=obj,error=error)
                db.execute('update attempts set state=?,payload=? where window=? and role=? and n=?',('valid' if obj else 'invalid',encode(data),r['window'],r['role'],n))
                if obj is None and n==1:retries.append((r,2))
        if failure and any(n==2 for r,n in batch):raise RuntimeError(failure)
        for item in retries:attempt([item])
    for i in range(0,len(todo),2):
        attempt(todo[i:i+2])
        if i%32==0 or i+2>=len(todo):print(time.strftime('%FT%T%z'),i+min(2,len(todo)-i),'/',len(todo),flush=True)
    save(EV/'teacher_finish.json',dict(at=time.time(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
if __name__ == "__main__":
    lock_sources(7)
    main()
