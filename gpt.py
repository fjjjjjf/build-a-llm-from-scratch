import torch 
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader
from MultiHeadAttention import MultiHeadAttention
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
'''
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
'''
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

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att= MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
        shortcut =x
        x=self.norm1(x)
        x=self.att(x)
        x=self.drop_shortcut(x)
        x=x+shortcut  #快捷连接
        shortcut =x 

        x=self.norm2(x)
        x=self.ff(x)
        x=self.drop_shortcut(x)
        x=x+shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()        
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'],cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg['emb_dim'])

        self.out_head =nn.Linear(
            cfg['emb_dim'],cfg['vocab_size']
        )

    def forward(self,in_idx):
        batch_size ,seq_len =  in_idx.shape
        tok_embeds =self.tok_emb(in_idx)
        pos_embeds =self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):  # idx 是当前上下文中索引的数组，形状为 (batch, n_tokens)
        idx_cond = idx[:,-context_size:] #若上下文长度超出支持范围，则进行裁剪。例如，若模型仅支持 5 个 token，而上下文长度为 10，仅使用最后 5 个 token 作为上下文 ,取每行最后5个
        with torch.no_grad():
            logits = model(idx_cond) 

        logits =logits[:,-1,:] #仅关注最后一个时间步，将形状从 (batch, n_token, vocab_size) 转换为 (batch, vocab_size)
        probas = torch.softmax(logits,dim=-1) #probas 的形状为 (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)       #idx_next 的形状为 (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)   #将采样的索引追加到当前序列中，此时 idx 的形状为 (batch, n_tokens+1)

    return idx

if __name__ =='__main__':
    tokenizer = tiktoken.get_encoding('gpt2') #使用openai的bpe分词器，后续可以自己弄 
    ''' 
    batch =[]
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2))) 
    batch= torch.stack(batch,dim=0) #堆叠batch
    batch=batch
    print('batch shape: ',batch.shape)

    torch.manual_seed(123)

    model =GPTModel(GPT_CONFIG_124M)
    out =model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)
    '''  
    model =GPTModel(GPT_CONFIG_124M)
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)            #添加批次维度
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()             # 禁用dropout
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )


    print("Output:", out)
    print("Output length:", len(out[0]))
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)