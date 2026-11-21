"""Complete the offline clone and perform phase-1 checks when transfers finish."""
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

env = Path(__file__).resolve().parent
run = env.parent
root = run.parents[2]
prefix = root / "runtime/envs/v2"
conda = "/opt/conda/bin/conda"
deadline = int((env / "start_epoch").read_text()) + 4*3600
def wait_for(check):
    while not check():
        if time.time() >= deadline:
            raise TimeoutError("phase-1 preparation timed out; see prerequisite job status")
        time.sleep(15)

def clone_finished():
    status = json.loads((run/"jobs/conda-clone/status.json").read_text())
    if status['state'] != 'running':
        return True
    if not Path('/proc',str(status['pid'])).exists():
        raise RuntimeError('clone worker missing without terminal state')
    return False

wait_for(clone_finished)
state = json.loads((run/"jobs/conda-clone/status.json").read_text())
if state['exit_code'] != 0 and not (env/'offline_recovery.txt').is_file():
    log = (run/"jobs/conda-clone/stage.log").read_text(errors='replace')
    if 'OfflineError' not in log and 'EnforceUnusedAdapter' not in log:
        raise RuntimeError('Inspect original clone error before recovery')
    archives = root / 'runtime/envs/phase1-archives'
    archives.mkdir(exist_ok=True)
    packages=json.loads((env/'conda-packages.json').read_text())
    with tarfile.open(root/'runtime/envs/phase1-conda-packages.tar') as tf:
        for row in packages:
            name=row['fn']
            assert Path(name).name == name
            f=tf.extractfile('./'+name)
            data=f.read()
            assert hashlib.sha256(data).hexdigest()==row['sha256'],name
            (archives/name).write_bytes(data)
    spec=env/'conda-local-explicit.txt'
    spec.write_text('@EXPLICIT\n'+'\n'.join((archives/r['fn']).as_uri() for r in packages)+'\n')
    (prefix/'conda-meta').mkdir(exist_ok=True)
    (prefix/'conda-meta/history').touch(exist_ok=True)
    subprocess.run([conda,'install','--offline','-y','--prefix',str(prefix),'--file',str(spec)],
                   env=dict(os.environ,CONDA_PKGS_DIRS=str(root/'runtime/envs/conda-cache')),check=True)
    (env/'offline_recovery.txt').write_text('Local explicit archives installed into clone after copying untracked pip files. See finish_preparation.py.\n')

(env/'conda-explicit.txt').write_bytes(subprocess.check_output([conda,'list','--explicit','--prefix',str(prefix)]))
python = str(prefix/'bin/python')
probe = "import torch,transformers; from transformers import Qwen3ForCausalLM; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
print(subprocess.check_output([python,'-B','-c',probe],text=True).strip(),flush=True)
source=json.loads((env/'source.json').read_text())
model=root/'model/Qwen3-8B'/source['revision']
wait_for(lambda: all((model/n).is_file() and (model/n).stat().st_size==r['size'] for n,r in source['files'].items()))
subprocess.run([python,str(root/'code/verify_phase1.py')],check=True)
