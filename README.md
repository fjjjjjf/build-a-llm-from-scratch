# build-a-llm-from-scratch 从零构建大语言模型

学习路径来自[
Build a Large Language Model (From Scratch) 中文版](https://skindhu.github.io/Build-A-Large-Language-Model-CN/#/)以及[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch/tree/main)

# 学习内容
### 0. [个人笔记](从零构建大模型.pdf)
### 1. 理解大语言模型
### 2. 处理文本数据
1. 将文本embedding化，即向量化
2. 可以使用gpt2的分词器，或者手写分词器，其中分词器实现的主要原理是bpe算法
3. 为了让llm理解文字位置，加入位置编码
4. 最终一个文本text最后转化为 text_embedding+position_embedding
### 3. 实现注意力机制
1. 注意力的重要性是为了让LLM能管住全部的词，不会因为前后词的距离而忘记，要重点看重之间的注意力
2. 自注意力机制的重点在于三个向量Q、K、V,
    1. query(查询向量代表了这个词在寻找相关信息时提出的问题)
    2. key(键向量代表了一个单词的特征，或者说是这个单词如何"展示"自己，以便其它单词可以与它进行匹配)
    3. value(值向量携带的是这个单词的具体信息，也就是当一个单词被"注意到"时，它提供给关注者的内容)
</p> 

3. 理解方式![](picture/1.png 'QKV生成方法')
4. 将得到的注意力得分进行归一化
5. 使用for循环计算所有的上下文向量，获得所有的注意力得分(也要进行归一化)
6. 大致计算过程![图片](picture/2.jpg)
7. 大致代码内容，但是实际要与矩阵的参数结合(batch, num_tokens, num_heads, head_dim)
    ```
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    
    keys = inputs @ W_key
    values = inputs @ W_value
    queries = inputs @ W_query

    attn_scores = values @ keys
    attn_weigths = softmax(...)

    context_vec = attn_weights @ values
8. 完整图例![1](picture\3.png)
9. 使用-INF 屏蔽后续词以及增加dropout来防止过拟合
### 4. 从零开始实现用于文本生成的GPT模型
### 5. 在无标记数据集进行预训练
### 6. 用于分类任务的微调
### 7. 指令遵循微调
