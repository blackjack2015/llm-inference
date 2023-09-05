import torch
from GPT_cpu import TransformerLMSequential,CacheAttention,CacheManager

import time

llama_params_7b = {
    "ntokens": 3200,
    "ninp": 4096,
    "nhead": 32,
    "nhid": 11008,
    "dropout": 0.1,
    "initrange": 0.02,
    "ndecoder":8,
    "use_cache":False
}

llama_params_30b = {
    "ntokens": 32000,
    "ninp": 7168,
    "nhead": 56,
    "nhid": 11008,
    "dropout": 0.1,
    "initrange": 0.02,
    "ndecoder":48,
    "use_cache":True
}

def create_llama_model():
    return TransformerLMSequential(**llama_params_7b)
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    total_params_in_billion = total_params / 1e9
    return total_params_in_billion   


model = create_llama_model()
# cache_manager = CacheManager(model)
total_params_in_billion = count_parameters(model)
print("Total number of model parameters (in billions):", total_params_in_billion)

src = torch.randint(1, (1, 16))
iters = 1500
total_time = 0
with torch.no_grad():
    for i in range(10):
        # warmup
        out = model(src)
    for i in range(iters):
        start = time.time()
        out = model(src)
        print('done')
        end = time.time()
        total_time += end - start
        # if i % 15 ==0:
        #     cache_manager.clear_cache()
print("average time:", total_time / iters)
