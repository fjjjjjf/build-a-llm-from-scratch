import json
import os
import urllib.request
import torch
from torch.utils.data import Dataset    
from torch.utils.data import DataLoader
import tiktoken
from gpt_download import download_and_load_gpt2
from gpt_generate import load_weights_into_gpt,GPTModel,text_to_token_ids,token_ids_to_text,generate,train_model_simple,calc_loss_loader
import time
from gpt_class_finetune import plot_values
from tqdm import tqdm
import re

class InstructionDataset(Dataset):
    def __init__(self,data,tokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data :
            instruction_plus_input = format_input(entry)
            response_text =f"\n\n### Response:\n{entry['output']}"
            full_text =instruction_plus_input+response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    def __getitem__(self, index):
        return self.encoded_texts[index]
    def  __len__(self):
        return len(self.data)
    

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    # 无论是新下载还是已存在，都只需要读一次
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task."
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry['input'] else ''

    return instruction_text+input_text

def custom_collate_draft_1(
        batch,
        pad_token_id=50256,
        device= 'cpu'
):  #填充不够长的token为50256
    batch_max_length =max(len(item)+1 for item in batch)
    inputs_lst= []

    for item in batch:
        new_item=item.copy()
        new_item+=[pad_token_id]

        padded =new_item+[pad_token_id]*(batch_max_length-len(new_item)) #进行填充

        inputs =torch.tensor(padded[:-1]) #删除之前多的填充token
        inputs_lst.append(inputs)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

def custom_collate_draft_2(
        batch,
        pad_token_id=50256,
        device= 'cpu'
): #为当前输入的token生成下一个target token
    batch_max_length =max(len(item)+1 for item in batch)
    inputs_lst,targets_lst =[],[]

    for item in batch:
        new_item = item.copy()
        new_item +=[pad_token_id]
        padded =new_item+[pad_token_id]*(batch_max_length-len(new_item))
        inputs =torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])  #在目标列表中仍保留一个50256，是为了llm学习在接受到指令时候何时结束token
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor,targets_tensor

def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length =None,
        device='cpu'
):
    batch_max_length =max(len(item)+1 for item in batch)
    inputs_lst,targets_lst =[],[]
    for item in batch:  
        new_item = item.copy()
        new_item +=[pad_token_id]
        padded =new_item+ [pad_token_id]*(batch_max_length-len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets =torch.tensor(padded[1:])

        mask =targets == pad_token_id   #在targets中，将除第一个以外地所有填充标志换成ignore_index，换成-100
        indices =torch.nonzero(mask).squeeze()  #targets = [11, 12, 50256] => mask = [False, False, True]=>indices = [2]
 
        if indices.numel()>1:
            targets[indices[1:]] =ignore_index  #如果有多个50256,则除第一个的都替换成-100,替换成-100是因为计算交叉熵会忽略标签为-100的目标，减少损失

        if allowed_max_length is not None:
            inputs =inputs[:allowed_max_length]
            targets =targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    inputs_tensor =torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor,targets_tensor

if __name__ =='__main__':
    file_path = "instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    #划分数据集
    train_portion =int(len(data)*0.85)
    test_portion = int(len(data)*0.1)

    val_portion =  len(data)-train_portion-test_portion

    train_data = data[:train_portion]
    val_data =data[train_portion:train_portion+test_portion]
    test_data = data[train_portion+test_portion:]

    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers =0
    batch_size =8

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding('gpt2')

    train_dataset = InstructionDataset(train_data,tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers = num_workers
    )

    val_dataset = InstructionDataset(val_data,tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn= custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset =InstructionDataset(test_data,tokenizer)
    test_loader =DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn= custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )


    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split()[-1].strip("()")
    settings,params = download_and_load_gpt2(model_size=model_size,models_dir='gpt2')
    model = GPTModel(BASE_CONFIG)

    load_weights_into_gpt(model,params)
    model.eval() 
    model.to(device)

    # input_text = format_input(val_data[0])
    # print(input_text,'1')
    # token_ids =generate(model=model,
    #                     idx = text_to_token_ids(input_text,tokenizer).to(device),
    #                     max_new_tokens=50,
    #                     context_size=BASE_CONFIG['context_length'],
    #                     eos_id=50256)
    # generated_text = token_ids_to_text(token_ids,tokenizer)
    # response_text = generated_text[len(input_text):].strip()

    # print(response_text)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,model,device,num_batches=5)
        val_loss = calc_loss_loader(val_loader,model,device,num_batches=5)

    print("Training loss:",train_loss)
    print("Validation loss:",val_loss)

    start_time = time.time()
    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0005,weight_decay=0.1)
    num_epochs =2

    train_losses ,val_losses,token_seen =train_model_simple(
        model,train_loader,val_loader,optimizer,device,
        num_epochs=num_epochs,eval_freq=5,eval_iter=5,
        start_context=format_input(val_data[0]),tokenizer=tokenizer
    )

    end_time =time.time()
    execution_time_minutes =(end_time-start_time)/60

    print(f"Training completed in {execution_time_minutes:.2f} minutes.")


    # for entry in test_data[:3]:
    #     input_text =format_input(entry)
    #     token_ids = generate(model=model,idx =text_to_token_ids(input_text,tokenizer).to(device),
    #                          max_new_tokens=256,
    #                          context_size= BASE_CONFIG["context_length"],
    #                          eos_id=50256)
    #     generated_text = token_ids_to_text(token_ids,tokenizer)
    #     response_text  =generated_text[len(input_text):].replace('###Response:','').strip()
    #     print(input_text)
    #     print(f"\nCorrect response:\n>> {entry['output']}")
    #     print(f"\nModel response:\n>> {response_text.strip()}")
    #     print('-----------------------------------------------')


    for i,entry in tqdm(enumerate(test_data),total=len(test_data)):
        input_text = format_input(entry)

        token_ids = generate(model=model,idx =text_to_token_ids(input_text,tokenizer).to(device),
                             max_new_tokens=256,
                             context_size= BASE_CONFIG["context_length"],
                             eos_id=50256)
        generated_text = token_ids_to_text(token_ids,tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response",'').strip()
        test_data[i]['model_response'] =response_text
    with open('instruction-data-with-response.json','w') as file:
        json.dump(test_data,file,indent=4) 
    print(test_data[0])

    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft-standalone.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")

    # epochs_tensor =torch.linspace(0,num_epochs,len(train_losses))
    # plot_values(epochs_tensor,token_seen,train_losses,val_losses)
    
