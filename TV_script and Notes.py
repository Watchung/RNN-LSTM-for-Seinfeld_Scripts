# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:47:16 2022

@author: Shafufu

注意 RNN model 并不一定都是用来预测下一个词的。也常而用于情感分类任务。这时的 X input 可能就是每一个用户的评论。
    就不能如下的产生batch data。而是每一个人的评论完整的作为一个 X input，这是就时 变长序列 RNN　问题了。　
    https://blog.csdn.net/kejizuiqianfang/article/details/100835528
    
    


pytorch中LSTM的细节分析理解  https://blog.csdn.net/shunaoxi2313/article/details/99843368  HL；好帖

学会区分 RNN 的 output 和 state https://zhuanlan.zhihu.com/p/28919765                    HL；好帖

Famous blog explaining LSTM http://colah.github.io/posts/2015-08-Understanding-LSTMs/

nn.RNN pytorch中RNN参数的详细解释 :  https://blog.csdn.net/lwgkzl/article/details/88717678（有误，看评论） 
                                    https://blog.csdn.net/qq_34706871/article/details/103302554
                                    https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    1. input_size 输入RNN的维度，比如说NLP中你需要把一个单词输入到RNN中，这个单词的编码是300维的，
        那么这个input_size就是300.这里的input_size其实就是规定了你的输入变量的维度。用f(wX+b)来类比的话，这里输入的就是X 一个 observation的维度。
    2. hidden_size是啥？和最简单的BP网络一样的，每个RNN的节点实际上就是一个BP嘛，包含输入层，隐含层，输出层。这里的hidden_size呢，
        你可以看做是隐含层中，隐含节点的个数
    3. 当然 以上是对于RNN某一个节点而言的，那么如何规定RNN的节点个数呢？
    
Initial value for hidden state and cell state    
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(), #HL For hidden state
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()) #HL For cell state



Loop the data in estimation:
    
    for batch_i, (inputs, labels) in enumerate(train_loader, 1): 也可以写作

    for batch_i, data in enumerate(trainloader, 0):
        inputs, labels = data   # x,y = (11,22)

    enumerate(train_loader, 1)  指 index starts from 1 for enumerate  https://discuss.pytorch.org/t/how-does-enumerate-trainloader-0-work/14410/1


hidden 的个数。与 sequence length无关， 与 层数 n_layer有关，与每层的hidden 个数有关， 为什么还与batch_size有关？
    Training的时候：
    因为对每一个 X input e.g. word[0:9]，做predict output的时候都需要 hidden units的initial value同时作为input for the first time step (t=1),即 word[0]，
    个数为 n_layer 乘  hidden_dim （即每层的hidden unit个数）。
    所以对一个batch的 X input predict，就需要 n_layer 乘  hidden_dim 乘 batch_size 个input values of hidden units。
    hidden 大致可以理解为 raw outputs 或者 feature extractions （然后再进入MLP 并 做softmax 分类 https://zhuanlan.zhihu.com/p/28919765），
    hidden 的 weights 在每一个time step 是公用的！（sequence length 就是 每个 X input 中 time step 的个数）
    
output 怎么计算出来的:
    https://zhuanlan.zhihu.com/p/28919765
    见下面 罗本 -- 拆 forward_back_prop

Tensor .tolist()
    [int_to_vocab[i] for i in inputs[0,:].tolist()]
    
拆开rnn run nn.LSTM or nn.Embedding, or nn.Linear 时需要再后面加 .cuda, 否则input 在cuda上 而 LSTM Embedding model 在CPU上 会报错

    Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
    https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
    
   
rnn.forward() 中 embed_otuput 的维度：    
    embed_output = embed(nn_input)  # HL: embed_output.size()  torch.Size([3, 10, 200])。 其中 nn_input时 [3,10],由于  Embedding Dimension = 200，
            也即每一个词都被embeded成200个维度的vector，由于 sequence_length是10，每一个词（数）都有自己的 embedded vector， 这10个embedded vector也就是分别作为
            10个time step 的  X input 
    lstm_output, hidden = lstm(embed_output,  hidden)  #HL: lstm_output.size() [3, 10, 12] ; hidden[0].size() 是 [n_layer,batch_size,hidden_dim]
    lstm_output = lstm_output.contiguous().view(-1, hidden_dim)  #HL: lstm_output.size() [30, 12]  这一步的目的是因为 nn.Linear 
                                                                      只take 2 dim 的 intput （*，in_features)，所以暂时捏成2维table。
    out = fc(lstm_output)           #HL: [30,12] 作为 input进入nn.Linear; out.size() [30, 21388]
    # reshape into (batch_size, seq_length, output_size)
    out = out.view(batch_size, -1, output_size) #HL: out.size() [3, 10, 21388] #HL 捏回3维table，以便于取出最后一个time step的output
    # get last batch  #HL: No, this is not get the last batch, this is to get the output only for the last time-step.即对第十个词之后的下一个词的预测
    out = out[:, -1]  #HL:  out[:, -1] 其实是 out[:,-1,:] 的缩写，即在第二个维度上去最后一个element
    # return one batch of output word scores and the hidden state
    return out, hidden


next(self.parameters()).data
    1. https://discuss.pytorch.org/t/what-does-next-self-parameters-data-mean/1458/2
    2. next(self.parameters()).new_zeros()是什么  https://blog.csdn.net/ccbrid/article/details/89354882


Why does keeping the hidden state of the previous sequence and using it as initial hidden state for our current sequence improve the learning?
    https://stackoverflow.com/questions/41633295/lstm-state-within-a-batch
    目前觉得唯一解释的通的说法是 :https://blog.csdn.net/shunaoxi2313/article/details/99843368
    为什么要有batch_first这个参数呢？常规的输入不就是(batch, seq_len, hidden_size)吗？而且参数默认为False，也就是它鼓励你第一维不是batch，更奇怪了。
    取pytorch官方的一个tutorial（chatbot tutorial）中的一个图. 左边是我们的常规输入（先不考虑hidden dim，每个数字代表序列中的一个词），
    右边是转置后，第一维成了max_length。我们知道在操作时第一维一般可视为“循环”维度，因此左边一个循环项是一个序列，无法同时经LSTM处理，
    而右边跨batch的循环项相当于当前time step下所有序列的当前词，可以并行过LSTM。（当然不管你是否batch_first它都是这么处理的，这个参数应该只是提醒一下这个trick）

    HL: As I understand, the RNN process the [[0], [1], [2] ] in parallel, with the initial hidden state, 
    generate the hidden states for next time step data [[1], [2], [3] ].  这个步骤，从0，1，2 到1，2，3 是所说的retain the hidden state. 
    然而9，10，11 到 10，11，12 这一步，需要shuffle = False, 否则不合理。 再想一想

    First batch:
    tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
            [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
            [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
    
    Second batch:
    tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])

    Tensorflow的LSTM的stateful的实现
    http://qiaowei.site/2018/03/25/Tensorflow%20LSTM%E7%9A%84stateful%E7%9A%84%E5%AE%9E%E7%8E%B0/
    "stateful的含义是在每个epoch内部的训练不同的batch时，前一个batch的输出state是后一个batch的initial state"
    "stateless就是对于每次的训练batch，cell state与hidden state都是初始化为0开始计算的。也就是说不同batch之间是没有任何关系的"
    大部分的应用都是stateless，因为前后训练的序列是不相关的，这也是大部分模型在训练前都会将训练数据进行shuffle，以保证喂给模型的数据的概率分布是均匀的。

    当序列A必须在序列B训练之前被训练时，往往这类应用是属于stateful的，即在训练序列B是，我们希望序列A的输出state可以喂进序列B的输入state，
    使得两个序列保持前后的关系。这在时间序列分析的应用中常见，
    比如：序列A是（t0, t1, t2）
    序列B是（t1, t2, t3）
    而时间轴正是t0, t1, t2, t3这样发生的
    
    
"""

import os
import torch
import numpy as np

os.chdir(r"C:\Users\Shafufu\Desktop\Huacheng Doc\HL Python Learning\Udacity\04 Deep Learning\Project_3_TV_scripts\Solution\Generate-TV-Scripts-master")

# load in data
import helper
data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)


view_line_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))


import problem_unittests as tests
from collections import Counter
def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    word_counts = Counter(text)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    
    # Alternatively
    #vocab = tuple(set(text))
    #int_to_vocab = dict(enumerate(vocab))
    #vocab_to_int = {ch: ii for ii, ch in int_to_vocab.items()}
    
    # return tuple
    return (vocab_to_int, int_to_vocab)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)



def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    tokens = {'.': '||Period||',
              ',': '||Comma||',
              '"': '||Quotation_Mark||',
              ';': '||Semi_Colon||',
              '!': '||Exclamation_Mark||',
              '?': '||Ques_Mark||',
              '(': '||Left_Paren||',
              ')': '||Right_Paren||',
              '-': '||Dash||',
              '\n': '||Return||'
            }
    #Alternatively
# =============================================================================
#     tokens_dict = { '.': '<PERIOD>',
#                        ',': '<COMMA>',
#                        '"': '<QUOTATION_MARK>',
#                        ';': '<SEMICOLON>',
#                        '!': '<EXCLAMATION_MARK>',
#                        '?': '<QUESTION_MARK>',
#                        '(': '<LEFT_PAREN>',
#                        ')': '<RIGHT_PAREN>',
#                        '-': '<DASH>',
#                        '?': '<QUESTION_MARK>',
#                        '\n': '<NEW_LINE>'}    
# =============================================================================
    return tokens

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# pre-process training data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()



"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')


from torch.utils.data import TensorDataset, DataLoader

# HL 下面的这个batch_data function 产生stateless性质的数据，即每个连个batch 的sequences 没有任何顺序
# HL 难道使用shuffle=False 不是更合理吗
def batch_data(words, sequence_length, batch_size): #each x = torch.Size([batch_size, sequence_length]) and y=torch.Size([batch_size])
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    feature_tensors = []
    target_tensors = []
    for i in range(len(words)-sequence_length):
        feature_tensors.append(words[i:i+sequence_length])
        target_tensors.append(words[i+sequence_length])
    data = TensorDataset(torch.tensor(feature_tensors), torch.tensor(target_tensors))
    data_loader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=batch_size)
    # return a dataloader
    return data_loader

# there is no test for this function, but you are encouraged to create
# print statements and tests of your own

# test dataloader

test_text = range(50)
t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

data_iter = iter(t_loader)
sample_x, sample_y = data_iter.next()

print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)
print(sample_y)


import torch.nn as nn

class RNN(nn.Module):  # this is the model that generated predited value based on inputs. the weights are to be trained in optimization step (forward_back_prop function) 
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        self.input_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.n_layers = n_layers
        # set class variables
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        # define model layers
    
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # nn_input [batch_size, seq_length]
        # hidden [n_layers, batch_size, hidden_dim]
        # out [batch_size, seq_length, output_size]  
        
        # TODO: Implement function   
        batch_size = nn_input.size(0)
        embed_output = self.embed(nn_input)
        lstm_output, hidden = self.lstm(embed_output,  hidden)
        lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)
        out = self.fc(lstm_output)
        # reshape into (batch_size, seq_length, output_size)
        out = out.view(batch_size, -1, self.output_size)
        # get last batch
        out = out[:, -1]
        # return one batch of output word scores and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU  #HL： for every epoch
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        weight = next(self.parameters()).data
        # initialize hidden state with zero weights, and move to GPU if available
        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(), #HL For hidden state
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()) #HL For cell state
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_rnn(RNN, train_on_gpu)






#Define forward and backpropagation


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param rnn: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # TODO: Implement Function
    
    # move data to GPU, if available
    if train_on_gpu:
        inp, target = inp.cuda(), target.cuda()
    hidden = tuple([each.data for each in hidden])
    # perform backpropagation and optimization
    rnn.zero_grad()
    output, h = rnn(inp, hidden) # this calls rnn.forward(inp, hidden)
    loss = criterion(output, target)
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()
    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), h

# Note that these tests aren't completely extensive.
# they are here to act as general checks on the expected outputs of your functions
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)



"""
DON'T MODIFY ANYTHING IN THIS CELL
"""

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1): #epoch_i=0
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size) # [i.shape for i in hidden]
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1): #pass
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn





# Data params
# Sequence Length
sequence_length = 10  # of words in a sequence
# Batch Size
batch_size = 128

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)
x, y = iter(train_loader).next()

# Training parameters
# Number of Epochs
num_epochs = 10
# Learning Rate
learning_rate = 0.001 

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)
# Output size
output_size = vocab_size
# Embedding Dimension
embedding_dim = 200
# Hidden Dimension
hidden_dim = 250
# Number of RNN Layers
n_layers = 2

# Show stats for every n number of batches
show_every_n_batches = 500


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""

# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
print('Model Trained and Saved')








===================================拆解====================================

# Try shuffle = False and see inputs
def batch_data(words, sequence_length, batch_size): #each x = torch.Size([batch_size, sequence_length]) and y=torch.Size([batch_size])
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    feature_tensors = []
    target_tensors = []
    for i in range(len(words)-sequence_length):
        feature_tensors.append(words[i:i+sequence_length])
        target_tensors.append(words[i+sequence_length])
    data = TensorDataset(torch.tensor(feature_tensors), torch.tensor(target_tensors))
    data_loader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=batch_size)
    # return a dataloader
    return data_loader





# Data params
# Sequence Length
sequence_length = 10  # of words in a sequence
# Batch Size
batch_size = 3  # 每次运算3组input

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)
x, y = iter(train_loader).next();x


tmp_loader = batch_data(range(10,6400), sequence_length, batch_size)
tmp_x, tmp_y = iter(tmp_loader).next(); tmp_x



# Training parameters
# Number of Epochs
num_epochs = 10
# Learning Rate
learning_rate = 0.001 

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)
# Output size
output_size = vocab_size # for every input, we predict one out of 21388 words
# Embedding Dimension
embedding_dim = 200
# Hidden Dimension
hidden_dim = 12
# Number of RNN Layers
n_layers = 2

# Show stats for every n number of batches
show_every_n_batches = 500


rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

n_epochs=1
# 拆 train_rnn==========================

# def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1): #epoch_i=0
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size) #HL [i.shape for i in hidden]  hidden[0].size() 每个维度分别对应 self.n_layers, batch_size, self.hidden_dim
        
        #for batch_i, (inputs, labels) in enumerate(train_loader, 1): #HL: batch_i 从 1开始是为了后面跟n_batch 比较，决定在哪里 break。所以不从0开始。
            batch_i=1;
            train_loader = batch_data(int_text, sequence_length, batch_size); 
            inputs, labels =  iter(train_loader).next() #HL get data for 拆解 运算
            #HL (x,y)=(111,222);   inputs.shape;  labels.size()
            #HL [int_to_vocab[i] for i in inputs[0,:].tolist()]
            #HL [int_to_vocab[i] for i in inputs[1,:].tolist()]
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size        #HL: len(train_loader.dataset) 对比  len(int_text)
            if(batch_i > n_batches):
                break;                                               #HL: 必须break，因为hidden.size() 是 [2, 128, 250]  而train_loader最后一份 data 的 inputs.size() 不足128 个 batch了。forward_back_prop 会计算报错 
            
            # forward, back prop   #HL 下一行code 见下面的拆解 forward_back_prop
            #loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)     #HL: hidden[0].size()      inputs.size()
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn



# 拆 forward_back_prop======================

# def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden): #hl  
    inp = inputs; target = labels
    """
    Forward and backward propagation on the neural network
    :param rnn: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # TODO: Implement Function
    
    # move data to GPU, if available
    if train_on_gpu:
        inp, target = inp.cuda(), target.cuda()
    #hl: hidden = rnn.init_hidden(batch_size)   #hidden 本身就是 两个elements的tuple，第一个element是h的initial value, 第二个是c的value
    hidden = tuple([each.data for each in hidden])  # 本行的目的是刷新一个hidden 这个tuple，否则 h[0].requires_grad  h[0].is_leaf 有问题
    # perform backpropagation and optimization
    rnn.zero_grad()
    
    #HL 下一行code 见下面拆解rnn
    #output, h = rnn(inp, hidden) # this calls rnn.forward(inp, hidden)
    loss = criterion(output, target)
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()
    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), h




# 拆 RNN ===========================================
dropout=0.5
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        #input_size = vocab_size
        #hidden_dim = hidden_dim
        #output_size = output_size
        #n_layers = n_layers
        # set class variables
        embed = nn.Embedding(vocab_size, embedding_dim).cuda()  #HL Note we need to add .cude() here
        lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True).cuda()
        fc = nn.Linear(hidden_dim, output_size).cuda()
        # define model layers
    
    
    def forward(self, nn_input, hidden):
        #HL  
        nn_input=inp
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
     
        
        # TODO: Implement function   #HL: 想清楚每一个步骤input 和 output的维度  
        batch_size = nn_input.size(0)   #HL: nn_input.size()  torch.Size([batch_size, seq_length])
        embed_output = embed(nn_input)  # HL: embed_output.size()  torch.Size([3, 10, 200]) 见 commens on the top
        lstm_output, hidden = lstm(embed_output,  hidden)  #HL: lstm_output.size() [3, 10, 12] ; hidden[0].size() 是 [n_layer,batch_size,hidden_dim]
        lstm_output = lstm_output.contiguous().view(-1, hidden_dim)  #HL: lstm_output.size() [30, 12]  这一步的目的是因为 nn.Linear 只take 2 dim 的 intput （*，in_features)，所以暂时捏成2维table。
        out = fc(lstm_output)           #HL: [30,12] 作为 input进入nn.Linear; out.size() [30, 21388]
        # reshape into (batch_size, seq_length, output_size)
        out = out.view(batch_size, -1, output_size) #HL: out.size() [3, 10, 21388] #HL 捏回3维table
        # get last batch
        out = out[:, -1]
        # return one batch of output word scores and the hidden state
        return out, hidden
    


































































































































































