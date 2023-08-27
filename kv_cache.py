import torch
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer


num_gpus = torch.cuda.device_count()
if num_gpus < 1:
    print("You need at least one GPU to run this notebook.")
    exit(0)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

models = [GPT2LMHeadModel.from_pretrained('gpt2').to(torch.device(f'cuda:{i}')) for i in range(num_gpus)]
devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
input_ids_list = [torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device) for device in devices]
past_key_values_list = [None] * num_gpus
streams = [torch.cuda.Stream(device=device) for device in devices]


start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_gpus)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_gpus)]

time_at_512 = [None] * num_gpus
mem_at_512 = [None] * num_gpus

timestamp_at_512 = [0] * len(devices)

for i in range(1, 512 + 1):
    for idx, (model, device, stream, input_ids, past_key_values, start_event, end_event) in enumerate(zip(models, devices, streams, input_ids_list, past_key_values_list, start_events, end_events)):
        with torch.cuda.stream(stream):
            if past_key_values is not None:
                past_key_values = tuple(tuple(past.to(device) for past in past_kv) for past_kv in past_key_values)

            outputs = model(input_ids, use_cache=True, past_key_values=past_key_values)

            past_key_values = tuple(tuple(past.cpu() for past in past_kv) for past_kv in outputs.past_key_values)
            input_ids = outputs.logits.argmax(-1)[:, -1].unsqueeze(-1).to(device)

            if i == 512:
                torch.cuda.synchronize(device)

                start_event.record()
                timestamp_at_512[idx] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') 
                past_key_values = tuple(tuple(past.to(device) for past in past_kv) for past_kv in past_key_values)
                end_event.record()
                torch.cuda.synchronize(device) 

                mem_at_512[idx] = torch.cuda.memory_allocated(device)
                time_at_512[idx] = start_event.elapsed_time(end_event)

for idx, (time, mem, timestamp) in enumerate(zip(time_at_512, mem_at_512, timestamp_at_512)):
    print(f'GPU {idx}: time at length 512 = {time} ms, memory at length 512 = {mem} B, timestamp = {timestamp}')
torch.cuda.synchronize()


