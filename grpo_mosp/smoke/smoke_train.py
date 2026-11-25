"""Offline CPU GRPO integration using a tiny randomly initialized Qwen2 + LoRA.

Exercises actual transformer gradients and adapter-disabled reference likelihoods.
This is a plumbing check, not evidence of pretrained Qwen task performance.
"""
import json
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
from ..policy import ConstrainedPolicy
from ..rollout import group
from ..grpo import update
from ..tests.helpers import ByteTokenizer, fixture


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config=Qwen2Config(vocab_size=256,hidden_size=32,intermediate_size=64,
                      num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
                      max_position_embeddings=16384,attention_dropout=0.0,sliding_window=None)
    model=get_peft_model(Qwen2ForCausalLM(config),LoraConfig(r=4,lora_alpha=8,
                        lora_dropout=0.0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.train()  # Zero configured dropout, with training checkpointing active.
    policy=ConstrainedPolicy(model,ByteTokenizer(),context_limit=16384)
    tasks=group(policy,fixture(),k=1,size=8,budget=1)
    assert len({r.reward for r in tasks})>1, 'Smoke seed must produce nonzero outcome variance'
    # Check exact stored old likelihoods before any optimizer step.
    with torch.no_grad():
        for r in tasks:
            for d in r.decisions:
                assert torch.allclose(policy.replay(d),torch.tensor(d.old_logps),atol=1e-5)
    parameters=[p for p in model.parameters() if p.requires_grad]
    before=[p.detach().clone() for p in parameters]
    optimizer=torch.optim.AdamW(parameters,lr=1e-3,weight_decay=0.0)
    report=update(policy,tasks,optimizer,dict(beta=.01,clip=.2,epochs_per_group=1,max_grad_norm=1))
    assert all(torch.isfinite(p).all() for p in parameters)
    assert any(not torch.equal(a,b) for a,b in zip(before,parameters))
    assert all(not r.env.failure for r in tasks)
    print(json.dumps(dict(test='tiny_random_Qwen2_LoRA_CPU',**report),indent=2))


if __name__=='__main__': main()
