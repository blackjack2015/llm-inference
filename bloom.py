from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

name = "bigscience/bloom-3b"
text = "Hello my name is"
max_new_tokens = 20

def generate_from_model(model, tokenizer):
    encoded_input = tokenizer(text, return_tensors='pt')
    output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda(), max_new_tokens=50)
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(name)

for i in range(100):
    result = generate_from_model(model, tokenizer)
    print(result)
