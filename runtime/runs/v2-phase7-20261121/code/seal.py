"""Seal stable artifacts after GPU and CPU workers both exit."""
import re,sys,subprocess
from prepare import *
def seal():
    for stage in ['prepare','teacher','handoff']:
        status=read(RUN/'jobs'/stage/'status.json')
        assert status['state']=='done' and status['exit_code']==0
    rc=read(RUN/'acceptance.json')
    assert rc['decisions']==1712 and rc['usable_demos']==sum(1 for _ in rows(D/'demos.jsonl'))
    report=(RUN/'phase7_acceptance.md').read_text()
    assert all((RUN/name).is_file() for name in re.findall(r'\]\(([^)]+)\)',report))
    paths=[p for folder in [D,EV,RUN/'code'] for p in folder.rglob('*') if p.is_file()]
    paths += [RUN/f for f in ['phase7_acceptance.md','acceptance.json','work_log.md','gpu_hours.csv']]
    for stage in ['prepare','teacher','handoff']:
        paths += [RUN/'jobs'/stage/n for n in ['status.json','stage.log']]
    sealfile=RUN/'phase7.sha256'
    sealfile.write_text(''.join(sha(p)+'  '+str(p.relative_to(RUN))+'\n' for p in sorted(paths)))
    subprocess.run(['sha256sum','-c','--status',sealfile.name],cwd=RUN,check=True)
    sealfile.chmod(0o444)
    print(len(paths),sha(sealfile),rc['status'])
if __name__=='__main__':seal()

