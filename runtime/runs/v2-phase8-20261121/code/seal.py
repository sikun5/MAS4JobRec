import subprocess
from common import *
def main():
    p=RUN/'jobs/pipeline/status.json'
    while read(p)['state']=='running':time.sleep(30)
    assert read(p)['state']=='done' and read(RUN/'acceptance.json')['status']=='passed'
    files=[p for stem in ['code','distill','student','eval','evidence'] for p in (RUN/stem).rglob('*') if p.is_file() and p.name!='resume.pt']
    files.extend(RUN/n for n in ['acceptance.json','acceptance_result_matrix.md','phase8_acceptance.md','work_log.md','gpu_hours.csv','manifest.json','resume_log.jsonl'])
    for j in (RUN/'jobs').iterdir():
        if j.name!='seal':files.extend(p for p in j.iterdir() if p.is_file() and p.name!='.lock')
    for n,h in read(E/'blind_seal.json')['files'].items():assert sha(RUN/n)==h
    manifest=RUN/'phase8.sha256';manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(RUN)}\n' for p in sorted(set(files))))
    subprocess.run(['sha256sum','-c','--status',manifest.name],cwd=RUN,check=True)
    manifest.chmod(0o444)
    print(len(set(files)),sha(manifest))
if __name__=='__main__':main()

