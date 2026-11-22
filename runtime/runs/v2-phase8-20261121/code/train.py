import random, math
from common import *
locked_access('train')
import torch,transformers,peft
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from transformers import AutoModelForCausalLM,AutoTokenizer

def main():
    demos=phase(7)/'distill/demos.jsonl'
    digest=sha(demos)
    assert digest=='8d78e98ec8b995a24046e503dbea8750e51fef870bbd455d832b0afcb2b00334'
    source=records(demos);assert len(source)==505 and len({(s['window'],s['role']) for s in source})==505
    torch.manual_seed(SEED);order=list(range(505));random.Random(SEED).shuffle(order)
    assert 'H100' in torch.cuda.get_device_name() and torch.cuda.device_count()==1
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'})
    model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules='all-linear',task_type='CAUSAL_LM'))
    model.train();model.enable_input_require_grads();model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,foreach=False)
    cfg=dict(base=str(BASE),demos_sha256=digest,role_conditioned=True,r=16,alpha=32,dropout=0.,target_modules='all-linear',trainable=sum(p.numel() for p in params),lr=1e-4,optimizer='AdamW',microbatch=1,accumulation=1,epochs=1,steps=505,seed=SEED,precision='bf16 base, fp32 adapters',torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,revision=BASE.name)
    write(D/'config.json',cfg)
    resume=D/'resume.pt';history=[]
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        assert ck['cfg']==cfg
        set_peft_model_state_dict(model,ck['weights']);opt.load_state_dict(ck['optimizer'])
        history=ck['history'];assert [r['source_index'] for r in history]==order[:len(history)]
        del ck
    for step,idx in enumerate(order[len(history):],len(history)+1):
        r=source[idx]
        prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        x=tok.encode(prompt,add_special_tokens=False);y=tok.encode(r['answer'],add_special_tokens=False)+[tok.eos_token_id]
        ids=torch.tensor([x+y],device='cuda');assert ids.numel()<=4096
        labels=ids.clone();labels[:,:len(x)]=-100
        t=time.monotonic();torch.cuda.reset_peak_memory_stats()
        loss=model(input_ids=ids,labels=labels,use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        norm=torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in params if p.grad is not None])).item()
        assert math.isfinite(norm)
        opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize()
        history.append(dict(step=step,source_index=idx,role=r['role'],window=r['window'],prompt_sha256=r['prompt_sha256'],prompt_tokens=len(x),answer_tokens=len(y),loss=loss.item(),grad_norm=norm,seconds=time.monotonic()-t,peak_allocated=torch.cuda.max_memory_allocated()))
        if step%25==0 or step==505:
            write_rows(D/'training.jsonl',history);print(step,history[-1]['loss'],flush=True)
        if step%100==0 or step==505:
            tmp=resume.with_suffix('.tmp')
            torch.save(dict(cfg=cfg,weights=get_peft_model_state_dict(model),optimizer=opt.state_dict(),history=history),tmp);tmp.replace(resume)
    assert len(history)==505 and all(x['grad_norm']>0 for x in history)
    model.save_pretrained(D/'adapter',safe_serialization=True)
    write(E/'adapter.json',{p.name:sha(p) for p in (D/'adapter').iterdir() if p.is_file()})
    log('共享适配器已保存，505 条示范各更新一次。')
if __name__=='__main__':main()

