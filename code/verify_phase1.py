"""Verify the deployed frozen snapshot, environment and preparation ledger on the HPC."""
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

root = Path(__file__).resolve().parents[1]
run = root / "runtime/runs/v2-phase1-20261121"
env = run / "env"
source = json.loads((env / "source.json").read_text())
model = root / "model/Qwen3-8B" / source["revision"]
assert source["model"] == "Qwen/Qwen3-8B"
assert Path(sys.prefix).resolve() == (root / "runtime/envs/v2").resolve()
checksums = []
for name, row in sorted(source["files"].items()):
    path = model / name
    assert path.stat().st_size == row["size"], name
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(block)
    assert digest.hexdigest() == row["sha256"], name
    checksums.append(f"{digest.hexdigest()}  {name}")
(env / "model_files.sha256").write_text("\n".join(checksums) + "\n")
index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
shards = sorted(set(index.values()))
assert len(shards) == 5
for name in shards:
    with safe_open(model / name, framework="pt", device="cpu") as f:
        assert set(f.keys()) == {k for k,v in index.items() if v == name}
config = AutoConfig.from_pretrained(model, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
assert config.model_type == "qwen3" and Qwen3ForCausalLM.__name__ in config.architectures
tokens = tokenizer.apply_chat_template([dict(role="user", content="岗位匹配")],
    tokenize=True, add_generation_prompt=True, enable_thinking=False)
assert len(tokens) > 0
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
card = torch.cuda.get_device_properties(0)
assert "H100" in card.name and card.total_memory >= 78 * 1024**3
assert torch.tensor([1], device="cuda").item() == 1
writable = []
for p in (root/"code", root/"data", run):
    with tempfile.NamedTemporaryFile(dir=p) as f:
        f.write(b"phase1"); f.flush()
        assert Path(f.name).read_bytes() == b"phase1"
    writable.append(str(p))
versions = {n:importlib.metadata.version(n) for n in ("torch","transformers","tokenizers","safetensors","accelerate")}
(env/"pip-freeze.txt").write_bytes(subprocess.check_output([sys.executable,"-m","pip","freeze","--all"]))
start = int((env/"start_epoch").read_text())
end = time.time()
utc = lambda t: datetime.fromtimestamp(t,timezone.utc).isoformat()
manifest = dict(run_id=run.name, project=str(root), paths={k:str(root/k) for k in ("code","data","global_guide","runtime","model")},
    writable_test=writable, interpreter=sys.executable, python=platform.python_version(), packages=versions,
    backend="Transformers batched generate", cuda_runtime=torch.version.cuda, device=card.name,
    vram_bytes=card.total_memory, driver=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).strip(),
    model=source["model"],revision=source["revision"],snapshot=str(model),file_count=len(checksums),verified_shards=shards,
    tensor_index_keys=len(index),offline_config=config.model_type,tokenizer=type(tokenizer).__name__,chat_template_test_tokens=len(tokens),
    env_rebuild="env/rebuild.md, env/conda-explicit.txt, env/pip-freeze.txt", stage_launcher="code/launch_stage.py",
    teacher_32b="Optional, not downloaded",pilot="not started",completed_utc=utc(end),
    filesystem_free_bytes=shutil.disk_usage(root).free,user_quota_command=shutil.which("quota"))
(env/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
with (run/"gpu_hours.csv").open("w") as f:
    writer=csv.writer(f); writer.writerow(["stage","start_utc","end_utc","gpu_hours","basis"])
    writer.writerow(["phase1",utc(start),utc(end),f"{(end-start)/3600:.6f}","conservative development-instance preparation interval; billing unverified"])
    writer.writerow(["pilot","","",0,"not started"])
print(json.dumps(manifest))
