"""Load the official AWQ GEMM checkpoint using AutoAWQ 0.2.9's Triton kernels."""
import importlib.util
import json
from experiment_paths import phase

import torch
from torch import nn
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

KERNEL = phase(3) / "pilot/vendor/autoawq-0.2.9/awq/modules/triton/gemm.py"
spec = importlib.util.spec_from_file_location("phase3_upstream_awq_gemm", KERNEL)
kernels = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = kernels
spec.loader.exec_module(kernels)


class AWQLinear(nn.Module):
    def __init__(self, linear, group):
        super().__init__()
        k,n=linear.in_features,linear.out_features
        self.register_buffer("qweight",torch.empty(k,n//8,dtype=torch.int32,device="meta"))
        self.register_buffer("qzeros",torch.empty(k//group,n//8,dtype=torch.int32,device="meta"))
        self.register_buffer("scales",torch.empty(k//group,n,dtype=torch.float16,device="meta"))
        self.bias=linear.bias

    def forward(self,x):
        x=x.to(torch.float16)
        flat=x.reshape(-1,x.shape[-1])
        if flat.shape[0]>=1024:
            result=flat@kernels.awq_dequantize_triton(self.qweight,self.scales,self.qzeros)
        else:
            result=kernels.awq_gemm_triton(flat,self.qweight,self.scales,self.qzeros,split_k_iters=8)
        if self.bias is not None:result=result+self.bias
        return result.view(*x.shape[:-1],-1)


def load_awq(src):
    config=AutoConfig.from_pretrained(src,local_files_only=True)
    q=config.quantization_config
    assert q["bits"]==4 and q["group_size"]==128 and q["version"]=="gemm" and q["zero_point"]
    with init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(config,dtype=torch.float16,attn_implementation="sdpa")
    weights=json.loads((src/"model.safetensors.index.json").read_text())["weight_map"]
    count=0
    for path in weights:
        if path.endswith(".qweight"):
            name=path.removesuffix(".qweight")
            parent,attr=name.rsplit(".",1)
            linear=model.get_submodule(name)
            setattr(model.get_submodule(parent),attr,AWQLinear(linear,q["group_size"]))
            count+=1
    assert set(model.state_dict())==set(weights),(set(model.state_dict())^set(weights))
    for shard in sorted(set(weights.values())):
        state=load_file(src/shard,device="cuda:0")
        # The checkpoint's unquantized tensors may be BF16. GEMM uses FP16.
        for k,v in list(state.items()):
            if v.is_floating_point():state[k]=v.half()
        model.load_state_dict(state,strict=False,assign=True)
        del state
    model=model.to("cuda")
    model.generation_config=GenerationConfig.from_pretrained(src,local_files_only=True)
    print("official AWQ GEMM modules",count,flush=True)
    return model.eval()
