import urllib.request
import zipfile
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
import tiktoken
from gpt_download import download_and_load_gpt2
from gpt_generate import load_weights_into_gpt,GPTModel,text_to_token_ids,token_ids_to_text,generate_text_simple

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

class SpamDataset(Dataset):
    def __init__(self,csv_file,tokenizer,max_len=None,pad_token_id=50256):
        self.data =pd.read_csv(csv_file)

        self.encoded_texts =[
            tokenizer.encode(text) for text in self.data['Text']
        ]
        if max_len is None:
            self.max_length  = self._longest_encoded_length()
        
        else:
            self.max_length= max_len
            self.encoded_texts= [
                encoded_text[:self.max_length]
                    for encoded_text in self.encoded_texts
            ]
        
        self.encoded_texts = [
            encoded_text +[pad_token_id]*(self.max_length-len(encoded_text))
                for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label =self.data.iloc[index]['Label']
        return (
            torch.tensor(encoded,dtype=torch.long),
            torch.tensor(label,dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)
       
    def _longest_encoded_length(self):
        max_length =0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length>max_length:
                max_length =encoded_length
        return max_length
        

def download_and_unzip_spam_data(url,zip_path,extracted_path,data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return 
    with urllib.request.urlopen(url) as response :  # 下载数据集
        with open(zip_path,'wb') as out_file:
            out_file.write(response.read())
    
    with zipfile.ZipFile(zip_path,'r') as zip_ref:  #解压数据集
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path)/"SMSSpamCollection"
    os.rename(original_file_path,data_file_path)    #为解压的数据集文件设置.csv文件扩展名

    print(f"File downloaded and saved as {data_file_path}")

def create_balanced_dataset(df):
    num_spam =df[df['Label']=='spam'].shape[0]   #统计垃圾短信的实例数量
    ham_subset = df[df['Label']=='ham'].sample(num_spam,random_state=123) #随机抽取正常邮件实例
    balanced_df = pd.concat([ham_subset,df[df['Label']=='spam']])  #将正常邮件和垃圾邮件合并
    return balanced_df

def random_split(df,train_frac,validation_frac):
    df =df.sample(frac=1,random_state =123 ).reset_index(drop=True)  #将整个dataframe 随机打乱

    train_end = int(len(df)*train_frac)   #获得训练的数量
    validation_end = train_end +int(len(df)*validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df,validation_df,test_df  #训练，验证，测试集

def calc_accuracy_loader(data_loader,model,device,num_batches=None):
    model.eval()
    correct_predictions,num_examples = 0,0

    if num_batches is None:
        num_batches=len(data_loader)
    else:
        num_batches =min(num_batches,len(data_loader))
    
    for i ,(input_batch,target_batch) in enumerate(data_loader):
        if i<num_batches:
            input_batch,target_batch = input_batch.to(device),target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:,-1,:]

            predicted_labels =torch.argmax(logits,dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions +=(predicted_labels==target_batch).sum().item()
        else:
            break
    return correct_predictions/num_examples


    
if __name__ =='__main__':
    #download_and_unzip_spam_data(url,zip_path,extracted_path,data_file_path)
    tokenizer = tiktoken.get_encoding('gpt2')

    
    # df = pd.read_csv(data_file_path,sep='\t',header= None,names=['Label','Text'])
    # balanced_df = create_balanced_dataset(df)
    

    # balanced_df['Label'] =balanced_df['Label'].map({"ham":0,"spam":1})  #转换token

    # train_df,validation_df,test_df= random_split(balanced_df,0.7,0.1)  #分割保存数据
    # train_df.to_csv("train.csv",index=None)
    # validation_df.to_csv("validation.csv",index=None)
    # test_df.to_csv("test.csv",index=None)


    train_dataset = SpamDataset(
        csv_file= 'train.csv',
        tokenizer= tokenizer,
        max_len= None,
    )

    val_dataset =SpamDataset(
        csv_file='validation.csv',
        tokenizer= tokenizer,
        max_len= train_dataset.max_length
    )

    test_dataset =SpamDataset(
        csv_file='test.csv',
        tokenizer= tokenizer,
        max_len= train_dataset.max_length
    )

    num_workers =0
    batch_size =8
    torch.manual_seed(123)

    train_loader =DataLoader(dataset= train_dataset,
                             batch_size=batch_size,
                             shuffle= True,
                             num_workers=num_workers,
                             drop_last= True
                             )
    val_loader =DataLoader(dataset= val_dataset,
                             batch_size=batch_size,
                             shuffle= True,
                             num_workers=num_workers,
                             drop_last= False
                             )
    test_loader =DataLoader(dataset= test_dataset,
                             batch_size=batch_size,
                             shuffle= True,
                             num_workers=num_workers,
                             drop_last= False
                             )
    
    for input_batch,target_batch in train_loader:
        pass
        
    
    # print ("Input batch dimension: ",input_batch.shape)  #[8,120]
    #print("Target batch dimension: ",target_batch.shape) #[8]


    ########################################
    #导入模型
    ########################################

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"

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

    '''
    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )
    '''
    model_size = CHOOSE_MODEL.split()[-1].strip("()")
    settings,params = download_and_load_gpt2(model_size=model_size,models_dir='gpt2')
    model = GPTModel(BASE_CONFIG)

    load_weights_into_gpt(model,params)
    model.eval() 

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=num_classes
    )
    #允许模型的最后一个transformer层和norm层参与训练
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    '''
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    '''
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(train_loader,model,device,num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader,model,device,num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader,model,device,num_batches=10)

    print(f'Training accuracy:{train_accuracy*100:.2f}%')
    print(f'Validation accuracy:{val_accuracy*100:.2f}%')
    print(f'Test accuracy:{test_accuracy*100:.2f}%')

