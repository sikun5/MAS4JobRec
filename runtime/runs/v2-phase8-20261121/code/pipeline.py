"""Keep stage transitions on the server across SSH disconnects."""
import subprocess
from common import *
def wait_stage(name):
    p=RUN/'jobs'/name/'status.json'
    while not p.exists() or read(p)['state']=='running':time.sleep(15)
    s=read(p);assert s['state']=='done' and s['exit_code']==0,(name,s)
def run_stage(name,script,gpu=False):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if gpu else '',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    subprocess.run([sys.executable,'code/launch_stage.py',str(RUN),name,'p8v1','--',sys.executable,'-u',str(RUN/'code'/script)],env=env,check=True)
    wait_stage(name)
if __name__=='__main__':
    wait_stage('train')
    for name,script,gpu in [('reviews','phase4_agent_reviews.py',True),('review-replay','verify_reviews.py',False),('blind','blind.py',False),('score','evaluate.py',False),('accept','accept.py',False)]:
        run_stage(name,script,gpu)

