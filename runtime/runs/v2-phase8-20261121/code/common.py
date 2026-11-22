import csv,hashlib,json,os,sys,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
ROOT=RUN.parents[2]
def phase(i):return RUN.parent/f'v2-phase{i}-20261121'
E=RUN/'evidence'
S=RUN/'student'
D=RUN/'distill'
OUT=RUN/'eval/student'
BASE=ROOT/'model/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218'
SEED=20261121
def read(p):return json.loads(p.read_text())
def records(p):
    with p.open() as f:return [json.loads(l) for l in f]
def encode(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def write(p,x):
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def write_rows(p,seq):
    tmp=p.with_suffix('.tmp')
    with tmp.open('w') as f:
        for x in seq:f.write(encode(x)+'\n')
    tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def log(msg):
    line=time.strftime('%FT%T%z')+'　'+msg
    with (RUN/'work_log.md').open('a') as f:f.write('\n- '+line+'\n')
    print(line,flush=True)
def locked_access(stage):
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,os.PathLike)):return
        path=os.path.abspath(args[0])
        if '/labels/' in path and not (stage=='calibrate' and path.endswith('/labels/C.jsonl') or stage=='score' and path.endswith('/labels/T.jsonl')):
            raise PermissionError(stage+' cannot open labels')
        if stage in ('train','review') and str(phase(6))+'/' in path:raise PermissionError(stage+' cannot open Phase 6 results')
    sys.addaudithook(guard)

