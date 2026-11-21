"""Verify the uploaded pinned teacher checkpoint, then run the three pilot stages serially."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
run=Path("runtime/runs/v2-phase3-20261121")
pilot=run/"pilot"
meta=json.loads((pilot/"hf.json").read_text())
folder=Path("model/Qwen3-32B-AWQ")/meta["sha"]
files={s["rfilename"]:s for s in meta["siblings"]}
while any(not (folder/n).is_file() or (folder/n).stat().st_size!=row["size"] for n,row in files.items()):
    time.sleep(15)
verified={}
for n,f in files.items():
 h=hashlib.sha256() if "lfs" in f else hashlib.sha1(b"blob "+str(f["size"]).encode()+b"\0")
 with (folder/n).open("rb") as r:
  for chunk in iter(lambda:r.read(8<<20),b""):h.update(chunk)
 assert h.hexdigest()==(f["lfs"]["sha256"] if "lfs" in f else f["blobId"]),n
 verified[n]=dict(size=f["size"],upstream_digest=h.hexdigest())
(pilot/"teacher_verified.json").write_text(json.dumps(dict(model=meta["modelId"],revision=meta["sha"],files=verified,verified_utc=datetime.now(timezone.utc).isoformat()),indent=2))
for mode in ["teacher","lora","full"]:
 print("starting",mode,flush=True)
 subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.free,utilization.gpu","--format=csv"],check=True)
 ret=subprocess.run([sys.executable,"code/launch_stage.py",str(run),mode,"probe1","--",sys.executable,"-u","code/phase3_probe.py",mode])
 ret.check_returncode()
 path=run/"jobs"/mode/"status.json"
 while True:
  time.sleep(5)
  if path.exists():
   state=json.loads(path.read_text())
   if state.get("state") in ("done","failed"):break
   try:os.kill(state["pid"],0)
   except ProcessLookupError:raise RuntimeError(f"stage owner missing: {mode}")
 print(mode,state["state"],flush=True)
 if mode=="teacher" and state["state"]!="done":raise RuntimeError("teacher failed")
