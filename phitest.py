import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Load the model and tokenizer with explicit float32 data type
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype=torch.float32)

prompt='''```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """'''

print("Input prompt:" + prompt)
print("-"*50) 

inputs = tokenizer('''```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

# Start timing
start_time = time.time()

outputs = model.generate(**inputs, max_length=200)

# End timing
end_time = time.time()

text = tokenizer.batch_decode(outputs)[0]
print(text)

# Calculate duration and tokens/sec
duration = end_time - start_time
total_tokens = inputs["input_ids"].shape[1] + outputs.shape[1]
tokens_per_second = total_tokens / duration

print("-"*50)

print(f"Processed {total_tokens} tokens in {duration:.2f} seconds. Rate: {tokens_per_second:.2f} tokens/sec.")
