import torch 
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class DummyGPTModel (nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  #Embeeding函数相当于查字典，根据位置查询对应向量
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks  =nn.Sequential(
            * [DummyTransformerBlock(cfg) for _ in range(cfg['n_layers'])] #为 TransformerBlock 设置占位符
        )
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])#为 LayerNorm 设置占位符
        self.out_head =nn.Linear(
            cfg['emb_dim'],cfg['vocab_size'],bias=False
        )

    def  forward(self,in_idx):
        batch_size,seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  
        pos_embdes= self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x=tok_embeds+pos_embdes
        x= self.drop_emb(x)
        x= self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):                                       #一个简单的占位类，后续将被真正的 TransformerBlock 替换
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):                                                     #该模块无实际操作，仅原样返回输入
        return x
    
class DummyLayerNorm(nn.Module):                                              #一个简单的占位类，后续将被真正的 DummyLayerNorm 替换
    def __init__(self, normalized_shape, eps=1e-5):                           # 此处的参数仅用于模拟LayerNorm接口
        super().__init__()

    def forward(self, x):
        return x
    
tokenizer = tiktoken.get_encoding('gpt2') #使用openai的bpe分词器，后续可以自己弄
batch =[]
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2))) 
batch= torch.stack(batch,dim=0) #堆叠batch
print('batch shape: ',batch.shape)

torch.manual_seed(123)
model =DummyGPTModel(GPT_CONFIG_124M)
logits =model(batch)
print('Output shape:',logits.shape)
print(logits)

#归一化学习
 
class  LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps =1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift =nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean =x.mean(dim=-1,keepdim =True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

def forward(self, x):
    return self.layers(x)