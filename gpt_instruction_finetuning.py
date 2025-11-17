import json
import os
import urllib.request
import torch
from torch.utils.data import Dataset    

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

        mask =targets == pad_token_id   #在targets中，将触底个意外地所有填充标志换成ignore_index，换成-100
        indices =torch.nonzero(mask).squeeze()

        if indices.numel()>1:
            targets[indices[1:]] =ignore_index

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
