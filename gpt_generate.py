import tiktoken
import torch
from gpt import GPT_CONFIG_124M,GPTModel,generate_text_simple

def text_to_token_ids(text, tokenizer):
    encode = tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encode_tensor =torch.tensor(encode).unsqueeze(0)
    return encode_tensor




if __name__ =='__main__' :
    text ='i as a'
    tokenizer = tiktoken.get_encoding('gpt2')

    print(text_to_token_ids(text=text,tokenizer=tokenizer))