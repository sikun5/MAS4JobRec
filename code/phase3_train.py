"""Probe one update per timed step with response-only role-conditioned SFT."""
import gc
import json
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM


def train(model,tok,mode,output,emit,template,inference):
    demos=json.loads((output/"teacher_demos.json").read_text())
    supplement=output/"teacher_supplement_demos.json"
    if supplement.exists():
        demos.extend(json.loads(supplement.read_text()))
    by_band={key:[r for r in demos if r["band"]==key] for key in ("typical","long")}
    assert all(by_band.values()),"no valid teacher data in a length band"
    model.train()
    model.config.use_cache=False
    if mode=="lora":
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,target_modules="all-linear",lora_dropout=0.,task_type="CAUSAL_LM"))
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    params=[v for v in model.parameters() if v.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,foreach=False)
    # A small monitored slice from a parameter with nonzero gradient establishes a real update.
    updates=[]
    for band,rows in by_band.items():
        for rep,role in enumerate(["skill","trajectory","preference","employer"]):
            matching=[r for r in rows if r["role"]==role]
            sample=matching[0] if matching else rows[rep%len(rows)]
            prompt=tok.encode(template(tok,sample["messages"]),add_special_tokens=False)
            answer=tok.encode(sample["answer"],add_special_tokens=False)+[tok.eos_token_id]
            ids=torch.tensor([prompt+answer],device="cuda")
            labels=ids.clone();labels[:,:len(prompt)]=-100
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            loss=model(input_ids=ids,labels=labels,use_cache=False).loss
            loss.backward()
            watch=next(p for p in params if p.grad is not None and p.ndim==2 and (p.grad[:8,:32]!=0).any())
            before=watch[:8,:32].detach().clone()
            grad=float(watch.grad[:8,:32].norm())
            opt.step()
            torch.cuda.synchronize();secs=time.monotonic()-start
            delta=float((watch[:8,:32].detach().float()-before.float()).abs().max())
            updates.append(delta)
            states=opt.state[watch]
            emit("train_step",band=band,rep=rep,query=sample["query"],window=sample["start"],role=sample["role"],seconds=secs,loss=float(loss),tokens=ids.numel(),supervised=len(answer),grad_norm=grad,
                 delta=delta,trainable=sum(p.numel() for p in params),parameters=sum(p.numel() for p in model.parameters()),
                 optimizer="AdamW foreach=False",state_dtype=str(states["exp_avg"].dtype),microbatch=1,accumulation=1,offload=False)
            assert grad>0 and torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
    assert max(updates)>0,"no observed parameter change"
    del opt,params,watch,states,loss,before
    model.gradient_checkpointing_disable()
    model.config.use_cache=True
    ckpt=output/(mode+"_checkpoint")
    t=time.monotonic()
    model.save_pretrained(ckpt,safe_serialization=True)
    emit("save",seconds=time.monotonic()-t)
    if mode=="lora":
        base=model.unload()
        del model
        gc.collect();torch.cuda.empty_cache()
        t=time.monotonic()
        model=PeftModel.from_pretrained(base,ckpt,is_trainable=False)
    else:
        # The full checkpoint is persisted. Re-evaluate the exact in-memory state.
        t=time.monotonic()
    emit("activate_student",seconds=time.monotonic()-t)
    model.eval()
    warm=by_band["typical"][0]["messages"]
    encoded=tok(template(tok,warm),return_tensors="pt",add_special_tokens=False).to("cuda")
    t=time.monotonic()
    with torch.inference_mode():
        model.generate(**encoded,do_sample=False,max_new_tokens=256,pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize();emit("warmup",seconds=time.monotonic()-t)
    inference(model,tok,["skill","trajectory","preference","employer"])
