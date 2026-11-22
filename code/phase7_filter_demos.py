"""Join frozen train labels only after the blind teacher completes."""
from experiment_paths import lock_sources
import csv,sqlite3
import json
from review_windows import SPECIALISTS
from phase7_prepare_windows import (D, P2, RUN, lines, read, rows)
from review_windows import parse
def filter_demos():
    status=read(RUN/'jobs/teacher/status.json')
    assert status['state']=='done' and status['exit_code']==0
    truth={r['query']:{o['slot']:o['J'] for o in r['outcomes']} for r in rows(P2/'labels/train.jsonl')}
    windows={w['window']:w for w in rows(D/'train_windows.jsonl')}
    db=sqlite3.connect('file:'+str(D/'attempts.sqlite')+'?mode=ro',uri=True)
    original=[];selected=[];decisions=[]
    reports={role:dict(role=role,logical=0,retries=0,valid=0,invalid=0,wrong_first=0,retained=0) for role in SPECIALISTS}
    people={role:set() for role in SPECIALISTS}
    for req in rows(D/'blind_requests.jsonl'):
        wid,q,role=(req[k] for k in ('window','query','role'))
        attempts=db.execute('select n,state,payload from attempts where window=? and role=? order by n',(wid,role)).fetchall()
        assert 1<=len(attempts)<=2
        report=reports[role];report['logical']+=1;report['retries']+=len(attempts)-1
        for i,(n,state,payload) in enumerate(attempts,1):
            a=json.loads(payload)
            assert (a['query'],a['window'],a['role'],a['attempt'],a['prompt_sha256'])==(q,wid,role,i,req['prompt_sha256'])
            assert state!='pending' and (i==len(attempts) or state!='valid')
            obj=parse(a['raw'],role)[0] if a['raw'] is not None else None
            assert obj==a['parsed'] and (state=='valid')==bool(obj)
            original.append(dict(**a,state=state))
        w=windows[wid]
        positive=next(j+1 for j,slot in enumerate(w['order']) if truth[q][slot])
        first=obj['rank'][0] if obj else None
        decision='invalid' if obj is None else ('retained' if first==positive else 'wrong_first')
        report['valid']+=int(obj is not None);report[decision]+=1
        decisions.append(dict(query=q,window=wid,role=role,attempts=len(attempts),decision=decision,positive_position=positive,teacher_first=first,prompt_sha256=req['prompt_sha256']))
        if decision=='retained':
            people[role].add(q)
            selected.append(dict(query=q,window=wid,role=role,messages=req['messages'],answer=a['raw'],prompt_sha256=req['prompt_sha256'],input_tokens=req['input_tokens'],output_tokens=a['output_tokens']))
    assert len(original)==db.execute('select count(*) from attempts').fetchone()[0]
    lines(D/'teacher_outputs.jsonl',original);lines(D/'filter_decisions.jsonl',decisions);lines(D/'demos.jsonl',selected)
    for role,rep in reports.items():
        rep.update(attempted_users=214,retained_users=len(people[role]),retention_rate=rep['retained']/rep['logical'])
    with (D/'role_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(reports['skill']),lineterminator='\n');w.writeheader();w.writerows(reports.values())
if __name__ == "__main__":
    lock_sources(7)
    filter_demos()
