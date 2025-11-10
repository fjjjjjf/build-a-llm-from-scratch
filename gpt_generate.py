import tiktoken
import torch
from gpt import GPT_CONFIG_124M,GPTModel,generate_text_simple
from MultiHeadAttention import create_dataloader_v1
import matplotlib.pyplot as plt
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny() #A
    ax2.plot(tokens_seen, train_losses, alpha=0) #B
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()
def text_to_token_ids(text, tokenizer):
    encode = tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encode_tensor =torch.tensor(encode).unsqueeze(0)
    return encode_tensor

def token_ids_to_text(token_ids, tokenizer):
    uncode =  token_ids.squeeze(0)
    
    return  tokenizer.decode(uncode.tolist()) #tolist,将tensor转换成list

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch =target_batch.to(device)

    logits = model(input_batch)
    loss =torch.nn.functional.cross_entropy(
        logits.flatten(0,1),target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader)==0:
        return float('nan')
    elif num_batches==None:
        num_batches=len(data_loader)
    else: 
        num_batches= min (num_batches,len(data_loader))
    print('batch: ',num_batches)
    for i ,(input,target) in enumerate(data_loader):
        if i<num_batches:
            loss=calc_loss_batch(input_batch=input,target_batch=target,model=model,device=device)
            total_loss+=loss.item()
        else: break
    return total_loss/num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):  #eval_iter评估时最多用多少 batch
    train_losses, val_losses, track_tokens_seen = [], [], []                #track_tokens_seen：记录当时训练了多少 tokens       
    tokens_seen, global_step = 0, -1 #token_seen记录记录训练过程中累计处理的 token 数量
    for epoch in range(num_epochs):    #global_step：记录当前训练步数
        model.train() #启用dropout
        for input_batch,target_batch in train_loader:

            optimizer.zero_grad()
            loss=calc_loss_batch(input_batch,target_batch,model,device) #前向传播
            loss.backward()  #反向传播
            optimizer.step()
            tokens_seen += input_batch.numel()  #numel()返回当前batch的token数量
            global_step += 1

            if global_step % eval_freq == 0:        #每隔多少freq进行模型评估
                train_loss, val_loss = evaluate_model(
                model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        generate_and_print_sample(       #生成一段文字查看效果                                            #每个 epoch 结束后打印示例文本
            model, tokenizer, device, start_context
        )
        
    return train_losses,val_losses,track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()                #评估阶段不用dropout
    with torch.no_grad():       #禁用梯度跟踪，减少计算开销
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()   #恢复
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=15,context_size=GPT_CONFIG_124M["context_length"],
            top_k=25,temperature=1.4
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " ")) # Compact print format
    model.train()

def generate(model,idx,max_new_tokens,context_size,
             temperature=1.0,top_k=None,eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits= model(idx_cond)
        logits = logits[:,-1,:]
        if top_k is not None:  #top_k 过滤
            top_logits ,_ =torch.topk(logits,top_k)
            min_val = top_logits[:,-1]
            logits =torch.where(
                logits<min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature >0.0 : #temperature 过滤
            logits =logits/temperature
            probs = torch.softmax(logits,dim=-1)
            idx_next =torch.multinomial(probs,num_samples=1)
        else:
            idx_next = torch.argmax(logits,dim=-1,keepdim=True)
        if idx_next ==eos_id:
            break
        #idx_next = idx_next.unsqueeze(1)
        idx =torch.cat((idx,idx_next),dim=1)
    return idx

if __name__ =='__main__' :
    torch.manual_seed(123)
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding('gpt2')
    model =GPTModel(GPT_CONFIG_124M)
  
    file_path ='the-verdict.txt'
    with open(file_path,'r',encoding='utf-8') as file:
        text_data = file.read()

    train_ratio = 0.9

    train_id = int(train_ratio*len(text_data))    
    train_data = text_data[:train_id]
    eval_data =text_data[train_id:]

    train_loader= create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length= GPT_CONFIG_124M['context_length'],
        stride= GPT_CONFIG_124M['context_length'],
        drop_last= True,
        shuffle=True,
        num_workers=0
    )

    eval_loader= create_dataloader_v1(
        eval_data,
        batch_size=2,
        max_length= GPT_CONFIG_124M['context_length'],
        stride= GPT_CONFIG_124M['context_length'],
        drop_last= False,
        shuffle=False,
        num_workers=0
    )

    device='cpu'
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)      #A
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, eval_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )
   
    torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
    )

    # epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    # plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
  