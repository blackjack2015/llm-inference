from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

def generate_text(prompt, output_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model = model.to(device)

    generated_ids = model.generate(input_ids,
                                   max_new_tokens=output_length,
                                   num_return_sequences=1,
                                   pad_token_id=tokenizer.eos_token_id,
                                   do_sample=True)

    print(input_ids.size())
    print(generated_ids.size())
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text

# 示例使用
prompt_text = "what is GPU DVFS?"
output_length = 100

start = time.time()
for i in range(10):
    results = generate_text(prompt_text, output_length)
print(results)
print((time.time() - start) / 10)

