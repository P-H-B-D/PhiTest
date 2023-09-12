import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer with explicit float32 data type
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype=torch.float32)

prompt='''Write me a story about the sun.'''

print("Input prompt:" + prompt)
print("-"*50) 

inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

# We'll generate tokens one by one
input_ids = inputs["input_ids"]
output = input_ids.tolist()[0]

# Let's set a limit for the number of tokens to generate to avoid infinite loops
max_tokens = 200

while len(output) < max_tokens:
    # Generate next token
    next_token_logits = model(input_ids).logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
    
    # Print the generated token
    next_token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
    print(next_token_text, end='', flush=True)
    
    # Append the token for the next iteration
    output.append(next_token.item())
    input_ids = torch.tensor([output], dtype=torch.int64)
