"""Allocation stress only, deliberately pad with pilot text up to the context cap."""
import time
import torch


def stress(model,tok,messages,emit):
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tok.encode(text,add_special_tokens=False)
    padding=tok.encode(messages[1]["content"],add_special_tokens=False)
    want=3840-len(ids)
    assert want>=0
    ids=ids[:12]+(padding*((want//len(padding))+1))[:want]+ids[12:]
    x=torch.tensor([ids],device="cuda")
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic()
    with torch.inference_mode():
        y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,
            min_new_tokens=256,max_new_tokens=256,pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    emit("stress",inputs=x.numel(),generated=y.shape[-1]-x.shape[-1],seconds=time.monotonic()-t,synthetic=True)
