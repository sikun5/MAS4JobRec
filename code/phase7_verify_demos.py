"""Replay window sampling and role decisions independently, then check student ingestion."""
from experiment_paths import ROOT, lock_sources
import csv,math,random,time
from collections import Counter,defaultdict
import hashlib
import itertools
import json
from review_windows import PROFILE, JOB, SPECIALISTS
from phase7_prepare_windows import (D, EV, P2, P3, SEED, encode, read, rows, save, sha)
from phase3_probe import TEACHER,BASE,prompt,template
from transformers import AutoTokenizer

def independent_parse(text,role):
    try:
        d=json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
        r=d['rank'];keys=['pJ']+{'preference':['pC'],'employer':['pE']}.get(role,[])
        assert isinstance(r,list) and len(r)==4 and set(r)=={1,2,3,4} and all(type(n)==int for n in r)
        for k in keys:
            assert isinstance(d[k],list) and len(d[k])==4 and all(type(p) in (float,int) and math.isfinite(p) and 0<=p<=1 for p in d[k])
        return {k:d[k] for k in ['rank']+keys}
    except (AssertionError,AttributeError,TypeError,ValueError,KeyError):
        return None
def verify():
    manifest=read(D/'teacher_manifest.json')
    for n,h in manifest['source_hashes'].items():assert sha(P2/n)==h
    for n,h in manifest['runtime_sources'].items():assert sha(ROOT/n)==h
    split=read(P2/'data/splits.json')
    train=split['standard']['train']
    users=sum(split['standard'].values(),[])
    for key in ['group_sha256','user_id']:assert len(users)==len({split['members'][q][key] for q in users})
    truths={x['query']:{o['slot']:o['J'] for o in x['outcomes']} for x in rows(P2/'labels/train.jsonl')}
    assert truths.keys()==set(train)
    lists={r['query']:r for r in rows(P2/'data/lists.jsonl') if r['query'] in truths}
    source={r['query']:r for r in rows(P2/'data/inputs.jsonl') if r['query'] in truths}
    windows=list(rows(D/'train_windows.jsonl'));wmap={w['window']:w for w in windows}
    expected=[];skipped=[]
    rng=random.Random(SEED);tok=AutoTokenizer.from_pretrained(TEACHER,local_files_only=True)
    stu=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    indexed={};shown={}
    def clip(f,cap):
        text=' '.join(key+'='+str(f[key]) for key in sorted(f,key=lambda k:k in {'experience','job_description'}))
        return tok.decode(tok.encode(text,add_special_tokens=False)[:cap],skip_special_tokens=True)
    for q in train:
        options=[c for c in itertools.combinations(sorted(truths[q]),4) if sum(truths[q][j] for j in c)==1]
        if not options:skipped.append(q)
        x=source[q];profile=clip(x['profile'],512)
        assert set(x['profile'])==set(PROFILE) and all(set(j['fields'])==set(JOB) for j in x['jobs'])
        jobs={j['slot']:clip(j['fields'],640) for j in x['jobs']}
        for i,c in enumerate(rng.sample(options,min(len(options),2)),1):
            order=list(c);rng.shuffle(order);wid=f'{q}-w{i}'
            expected.append(dict(query=q,window=wid,combination=list(c),order=order,list_sha256=lists[q]['sha256']))
            indexed[wid]=next(i for i,s in enumerate(order,1) if truths[q][s])
            for role in SPECIALISTS:shown[wid,role]=prompt(tok,profile,[jobs[j] for j in order],role)
    assert expected==windows and skipped==[r['query'] for r in rows(EV/'skipped.jsonl')]
    assert len(shown)==1712
    requests=list(rows(D/'blind_requests.jsonl'));assert len(requests)==len(shown)
    key=lambda r:(r['window'],r['role'])
    req={key(r):r for r in requests}
    assert req.keys()==shown.keys()
    for k,msg in shown.items():
        r=req[k]
        assert msg==r['messages'] and hashlib.sha256(encode(msg).encode()).hexdigest()==r['prompt_sha256']
        assert len(tok.encode(template(tok,msg),add_special_tokens=False))==r['input_tokens']
    attempts=defaultdict(list)
    for a in rows(D/'teacher_outputs.jsonl'):
        k=key(a); attempts[k].append(a)
        assert a['query']==req[k]['query'] and a['prompt_sha256']==req[k]['prompt_sha256']
        parsed=independent_parse(a['raw'],k[1])
        assert parsed==a['parsed'] and bool(parsed)==(a['state']=='valid')
        assert a['input_tokens']==req[k]['input_tokens'] and (a['output_tokens'] is None or 0<=a['output_tokens']<=256)
    assert attempts.keys()==req.keys()
    ds=list(rows(D/'filter_decisions.jsonl')); decisions={key(r):r for r in ds}
    assert len(ds)==len(decisions)==len(req) and decisions.keys()==req.keys()
    demos=list(rows(D/'demos.jsonl'));demo={key(x):x for x in demos};assert len(demos)==len(demo)
    counts=defaultdict(Counter);people=defaultdict(set);lengths=[]
    for k,aa in attempts.items():
        wid,role=k;c=counts[role];a=aa[-1];dec=decisions[k]
        assert [i['attempt'] for i in aa] in ([1],[1,2])
        assert all(i['parsed'] is None for i in aa[:-1])
        obj=a['parsed'];first=obj['rank'][0] if obj else None
        keep=first==indexed[wid]
        outcome='invalid' if not obj else ('retained' if keep else 'wrong_first')
        assert dec['teacher_first']==first and dec['positive_position']==indexed[wid] and dec['decision']==outcome and dec['attempts']==len(aa)
        assert (k in demo)==keep
        c.update(logical=1,retries=len(aa)-1,valid=int(bool(obj)));c[outcome]+=1
        if keep:
            z=demo[k];msg=shown[k]
            assert z['answer']==a['raw'] and z['messages']==msg and z['query']==wmap[wid]['query'] and z['prompt_sha256']==req[k]['prompt_sha256']
            people[role].add(z['query'])
            size=len(stu.encode(template(stu,msg),add_special_tokens=False))+len(stu.encode(z['answer'],add_special_tokens=False))+1
            assert size<=4096
            lengths.append(size)
    with (D/'role_results.csv').open() as f:reps=list(csv.DictReader(f))
    assert len(reps)==4 and {x['role'] for x in reps}==set(SPECIALISTS)
    for row in reps:
        role=row['role'];c=counts[role]
        for n in ['logical','retries','valid','invalid','wrong_first','retained']:assert int(row[n])==c[n]
        assert int(row['retained_users'])==len(people[role]) and int(row['attempted_users'])==214
        assert math.isclose(float(row['retention_rate']),c['retained']/c['logical'])
    assert read(P3/'audit.json')['three_routes_verified'] and (P3/'lora_checkpoint/adapter_model.safetensors').stat().st_size>0
    save(EV/'verified.json',dict(window_replay=len(windows),verified_prompts=len(req),raw_attempts=sum(map(len,attempts.values())),decisions=len(ds),usable_demos=len(demos),retained_unique_users=len(set.union(set(),*people.values())),student_token_range=[min(lengths),max(lengths)] if lengths else [],lora_path_verified=True,role_statistics=reps))
if __name__ == "__main__":
    lock_sources(7)
    verify()
