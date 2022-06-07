# -*- coding: utf-8 -*-
"""
https://blog.csdn.net/shunaoxi2313/article/details/99843368 

First question: what are the outputs of nn.RNN or nn.LSTM, and the meaning of them
    see Outputs: output, h_n from https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    The output of nn.RNN or nn.LSTM are processed features, which still needs to be input to MLP, softmax to get probability for each possible value.

Second question: what are the input and output data looks like. 
        x1=words[0:9], y1=words[10]; x2=words[1:10],y2=words[11]

 “The number of units in the hidden layers is basically the number of features the network can detect”。 
 https://classroom.udacity.com/nanodegrees/nd101/parts/cd0657/modules/d11f458a-086f-4814-8446-d56f8fd72bc3/lessons/3ccfc13a-c472-4146-a772-f4e1da5c43f4/concepts/97561b4f-a85b-48a5-8615-f6492a3183c7
        Tips and Tricks
        Monitoring Validation Loss vs. Training Loss
        If you're somewhat new to Machine Learning or Neural Networks it can take a bit of expertise to get good models. 
        The most important quantity to keep track of is the difference between your training loss (printed during training) 
            and the validation loss (printed once in a while when the RNN is run on the validation data (by default every 1000 iterations)). 
        
        In particular:
        If your training loss is much lower than validation loss then this means the network might be overfitting. 
        Solutions to this are to decrease your network size, or to increase dropout. For example you could try dropout of 0.5 and so on.
        If your training/validation loss are about equal then your model is underfitting. Increase the size of your model (either number of layers or the raw number of neurons per layer)
       
        Approximate number of parameters
        The two most important parameters that control the model are n_hidden and n_layers. I would advise that you always use n_layers of either 2/3. 
        The n_hidden can be adjusted based on how much data you have. The two important quantities to keep track of here are:
            The number of parameters in your model. This is printed when you start training.
            The size of your dataset. 1MB file is approximately 1 million characters.
        These two should be about the same order of magnitude. It's a little tricky to tell. Here are some examples:
        
            I have a 100MB dataset and I'm using the default parameter settings (which currently print 150K parameters). My data size is significantly larger (100 mil >> 0.15 mil), so I expect to heavily underfit. I am thinking I can comfortably afford to make n_hidden larger.
            I have a 10MB dataset and running a 10 million parameter model. I'm slightly nervous and I'm carefully monitoring my validation loss. If it's larger than my training loss then I may want to try to increase dropout a bit and see if that helps the validation loss.
       
        Best models strategy
        The winning strategy to obtaining very good models (if you have the compute time) is to always err on making the network larger (as large as you're willing to wait for it to compute) and then try different dropout values (between 0,1). Whatever model has the best validation performance (the loss, written in the checkpoint filename, low is good) is the one you should use in the end.
        
        It is very common in deep learning to run many different models with many different hyperparameter settings, and in the end take whatever checkpoint gave the best validation performance.
        
        By the way, the size of your training and validation splits are also parameters. Make sure you have a decent amount of data in your validation set or otherwise the validation performance will be noisy and not very informative.

梯度消失与梯度爆炸 Vanishing Gradient :
            https://zhuanlan.zhihu.com/p/25631496
           李宏毅     https://www.cnblogs.com/XDU-Lakers/p/10553239.html#:~:text=%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%EF%BC%9A%E5%9C%A8%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C,%E7%A7%8D%E7%8E%B0%E8%B1%A1%E5%8F%AB%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8%E3%80%82
            对于更普遍的梯度消失问题，可以考虑一下三种方案解决：
            （1）用ReLU、Leaky-ReLU、P-ReLU、R-ReLU、Maxout等替代sigmoid函数。(几种激活函数的比较见我的博客)
            （2）用Batch Normalization。(对于Batch Normalization的理解可以见我的博客)
            （3）LSTM的结构设计也可以改善RNN中的梯度消失问题。
            Long Short-Term Memory Cells (LSTMs) and Gated Recurrent Units (GRUs) give a solution to the vanishing gradient problem
            
每个mini batch 都必须 update parameters吗，可不可以可以减少更新次数：   pytorch里巧用optimizer.zero_grad增大batchsize    
            https://zhuanlan.zhihu.com/p/114435156
            1. 常规情况下，每个batch需要调用一次optimizer.zero_grad函数，把参数的梯度清零
                for i, minibatch in enumerate(tr_dataloader):
                    features, labels = minibatch
                    optimizer.zero_grad()
                    loss = model(features, labels)
                    loss.backward()
                    optimizer.step()
            2. 也可以多个batch 只调用一次optimizer.zero_grad函数。这样相当于增大了batch_size 
                for i, minibatch in enumerate(tr_dataloader):
                    features, labels = minibatch
                    loss = model(features, labels)
                    loss.backward()
                    if 0 == i % N:
                        optimizer.step()
                        optimizer.zero_grad()
                
LSTM summary: https://web.archive.org/web/20190106151528/https://skymind.ai/wiki/lstm


# Check the input and output dimensions
        print('Input size: ', test_input.size())
        # test out rnn sizes
        test_out, test_h = test_rnn(test_input, None)
        print('Output size: ', test_out.size())
        print('Hidden state size: ', test_h.size())
        Input size:  torch.Size([1, 20, 1]) # batch number, sequence (similar to depth in CNN), input size
        Output size:  torch.Size([20, 1]) # batch number x sequence, output size
        Hidden state size:  torch.Size([2, 1, 10]) # number of layers, batch size, hidden layer size (number of nodes)

# LSTM模型结构的图示 https://posts.careerengine.us/p/5fc0509003c7681ad8236c7c 

# ndarray.flatten https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
            a = np.array([[1,2], [3,4]])
            a.flatten()
            array([1, 2, 3, 4])
            a.flatten('F')
            array([1, 3, 2, 4])


# one_hot.reshape((*arr.shape, n_labels))   ; (*arr.shape, n_labels)


# Use yield to create generation function 
       1.  https://blog.csdn.net/mieleizhi0522/article/details/82142856
        到这里你可能就明白yield和return的关系和区别了，带yield的函数是一个生成器，而不是一个函数了，
        这个生成器有一个函数就是next函数，next就相当于“下一步”生成哪个数，这一次的next开始的地方是
        接着上一次的next停止的地方执行的，所以调用next的时候，生成器并不会从foo函数的开始执行，
        只是接着上一步停止的地方开始，然后遇到yield后，return出要生成的数，此步就结束。
        原文链接：https://blog.csdn.net/mieleizhi0522/article/details/82142856
            def foo():
                print("starting...")
                n=5
                while True:
                    res = yield n
                    print("res:",res)
                    n +=1
            g = foo()
            print(next(g))
            print("*"*20)
            print(next(g))
            
       2. https://zhuanlan.zhihu.com/p/268605982
        使用了yield的函数被称为生成器.在调用生成器函数的过程中，每次遇到 yield 时
        函数会暂停并保存当前所有的运行信息（保留局部变量），返回yield的值, 
        并在下一次执行next()方法时从当前位置继续运行，直到生成器被全部遍历完
        
# PyTorch - What does contiguous() do?
        https://stackoverflow.com/questions/48915810/pytorch-what-does-contiguous-do#
        https://blog.csdn.net/kdongyi/article/details/108180250
        https://blog.csdn.net/gdymind/article/details/82662502
        x = torch.randn(3, 2)
        y = torch.transpose(x, 0, 1)
        print("修改前：");y[0, 0] = 999;print("x-", x);print("y-", y)
        
        x = torch.randn(3, 2)
        y = torch.transpose(x, 0, 1).contiguous()
        print("修改后：");y[0, 0] = 999;   print("x-", x);print("y-", y)  #HL  .contiguous()相当于 .copy()

# LSTM: hidden unit, LSTM cell (layer) 
    0. https://jishuin.proginn.com/p/763bfbd33015
    1. https://stats.stackexchange.com/questions/241985/understanding-lstm-units-vs-cells
            Ah I see, so then a "cell" is a num_unit sized horizontal array of interconnected 
            LSTM state units. Makes sense. So then it would be analogous to a hidden layer 
            in a standard feed-forward network
    
    2. https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-an-lstm
    3. https://www.kaggle.com/andradaolteanu/pytorch-rnns-and-lstms-explained-acc-0-99
    
    总结：https://docs.google.com/document/d/1J6poGCZTXoutUC7nL1fkb0jKY5vy0I8MtSfrdZ7Xpn8/edit
    
# 无论是RNN 还是 LSTM， hidden state input dimension都是 n_layers, batch_size, n_hidden：
        containing the initial hidden state for each element in the batch. 
        For LSTM， Defaults to zeros if (h_0, c_0) is not provided
        #HL 每个input observation都需要自己的 initial hidden state value, 所以有下面的结论：

        ***注意*** batch size不影响 n_hidden unit 的个数，但是影响 n_hidden unit 的 
        initial values 的个数，which is 二者相乘。其实还应该再乘以n_layers。 注意区分
        
        为什么hidden initial values 需要两个 weight.new ? 下面有答案
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        
        https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
        
# What is the purpose of weight = next(net.parameters()).data  
        https://knowledge.udacity.com/questions/126503   
        https://knowledge.udacity.com/questions/260625
        By using .new we are constructing weights tensor having same data type to avoid any datatype conflicts
 
#  tensor.long()  loss = criterion(output, targets.view(batch_size*seq_length).long())  
            # self.long() is equivalent to self.to(torch.int64). See to()
            # https://pytorch.org/docs/stable/generated/torch.Tensor.long.html
            等同于 tensor.to(torch.float64)
            
#  h = tuple([each.data for each in h]) # 可以去掉这一行吗  不行！！
        RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed
        
        问题的完整表述：如果去掉 h = tuple([each.data for each in h])， epoch=0 可以正常运行，但是
        epoch=1 是，由于 上一个loop 中的 h[0].requires_grad = True, 所以 epoch=1 的backward 会 
        back propergate到 epoch=0的 计算，而此时epoch=0的graph已经被释放了。所以会 runtime error.
        具体的就是 在 epoch=0 的 output, h = net(inputs, h) 之后，  h[0].is_leaf is False。 将
        这个 h input到下一个 epoch 的  output, h = net(inputs, h) 后，backward 会试图 back prop
        到 h[0] 为leaf 的 step， 即上一个epoch。
        
        
        方法1： 去掉上一个loop中 h中的 requires_grad:
        http://blog.prince2015.club/2018/09/02/pytorch-twice-backward/
        https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
        使用 hidden.detach_() or hidden = hidden.detach()， 这样 h 中从上一个batch 的 requires_grad 就去掉了。
            h = tuple(each.data for each in h)
            output, h = net(inputs, h)
            loss = criterion(output, targets.view(batch_size*seq_length).long())
        这个epoch 就只需要back propergate 至目前的 h 值  因为就 input 到 net(inputs, h)【即graph生成的step】 
        中的 h[0].is_leaf = True。
        
        方法2： 保留之前各epoch生成的graph，就不再需要将 h 变成 leaf
        https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time-while-using-custom-loss-function/12360/2
        replace loss.backward() with loss.backward(retain_graph=True) 
        but know that each successive batch will take more time than the previous one because 
        it will have to back-propagate all the way through to the start of the first batch.
        
        一个关于 autograd & graph 非常饿清楚的图示，以及为什么需要optimizer.zero-grad()
        https://www.youtube.com/watch?v=MswxJw-8PvE
        
        
        
"""








import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

plt.figure(figsize=(8,5))

# how many time steps/data pts are in one batch of data
seq_length = 20
# generate evenly spaced data pts
time_steps = np.linspace(0, np.pi, seq_length + 1)  #HL time_step here is the seq_length
data = np.sin(time_steps)
data.resize((seq_length + 1, 1)) # size becomes (seq_length+1, 1), adds an input_size dimension
x = data[:-1] # all but the last piece of data
y = data[1:] # all but the first
# display the data
plt.plot(time_steps[1:], x, 'r.', label='input, x') # x
plt.plot(time_steps[1:], y, 'b.', label='target, y') # y
plt.legend(loc='best')

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim=hidden_dim
        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)  #############HL Note, this row gives model parameters, not model inputs!!
        # last, fully-connected layer                                                                 
        self.fc = nn.Linear(hidden_dim, output_size)
    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)  #HL time_step here is the seq_length
        batch_size = x.size(0)
        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)                  ##################HL: this is model with inputs for forward function
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)  # HL: flattenning step that returns batch x sequence number of row, and each columns for each hidden node
        # get final output 
        output = self.fc(r_out)
  
        return output, hidden

# Check the input and output dimensions
# test that dimensions are as expected
test_rnn = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2)  ##################HL: create model object, without input x yet
# generate evenly spaced, test data pts
time_steps = np.linspace(0, np.pi, seq_length)
data = np.sin(time_steps)
data.resize((seq_length, 1))

test_input = torch.Tensor(data).unsqueeze(0) # give it a batch_size of 1 as first dimension
print('Input size: ', test_input.size())

# test out rnn sizes
test_out, test_h = test_rnn(test_input, None) ################HL：这步其实是test_rnn.forward 的简写。
print('Output size: ', test_out.size())
print('Hidden state size: ', test_h.size())

'''
Input size:  torch.Size([1, 20, 1])
Output size:  torch.Size([20, 1])
Hidden state size:  torch.Size([2, 1, 10])
'''

# decide on hyperparameters
input_size=1 
output_size=1
hidden_dim=32
n_layers=1
# instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

# MSE loss and Adam optimizer with a learning rate of 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01) 

# train the RNN
def train(rnn, n_steps, print_every):
    
    # initialize the hidden state
    hidden = None      
    
    for batch_i, step in enumerate(range(n_steps)): #pass
        # defining the training data 
        time_steps = np.linspace(step * np.pi, (step+1)*np.pi, seq_length + 1)
        data = np.sin(time_steps)
        data.resize((seq_length + 1, 1)) # input_size=1

        x = data[:-1]
        y = data[1:]
        
        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0) # unsqueeze gives a 1, batch_size dimension
        y_tensor = torch.Tensor(y)

        # outputs from the rnn
        prediction, hidden = rnn(x_tensor, hidden)

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)
        # zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        # display loss and predictions
        if batch_i%print_every == 0:        
            print('Loss: ', loss.item())
            plt.plot(time_steps[1:], x, 'r.') # input
            plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.') # predictions
            plt.show()
    
    return rnn

# train the rnn and monitor results
n_steps = 75
print_every = 15
trained_rnn = train(rnn, n_steps, print_every)





=================== Character-Level LSTM in PyTorch ======================
import os, sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
os.chdir(r"C:\Users\Shafufu\Desktop\Huacheng Doc\HL Python Learning\Udacity\04 Deep Learning\Project_3_TV_scripts\RNN\char-rnn")

# open text file and read in data as `text`

with open('data/anna.txt', 'r') as f:
    text = f.read()
#text[:100]




#=======================Tokenization
# encode the text and map each character to an integer and vice versa
# we create two dictionaries:
# 1. int2char, which maps integers to characters
# 2. char2int, which maps characters to unique integers
chars = tuple(set(text)); print([i for i in chars])   # Note: chars is the Token
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
# encode the text
encoded = np.array([char2int[ch] for ch in text])
#encoded[:100]

def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot

# check that the function works as expected
test_seq = np.array([3, 5, 1])
one_hot = one_hot_encode(arr=test_seq, n_labels=8)
print(one_hot)
arr=encoded; arr.shape
batch_size = 100
seq_length = 100

#===============================Making training mini-batches
def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''
    
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]   # arr.shape (1980000, )
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    #arr.reshape((batch_size, -1,83)).shape
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length): #pass
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

batches = get_batches(encoded, 8, 50) #HL batches is a generator function
x, y = next(batches)


# printing out the first 10 items in a sequence
print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])


# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')


class CharRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        ## TODO: define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, #HL: 因为有83个character， 所以one hot encode of each character 长度也是83, 即 input_size = len(tokens) 
                            dropout=drop_prob, batch_first=True)
        
        ## TODO: define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))
      
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        ## TODO: Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        
        ## TODO: pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)
        
        ## TODO: put x through the fully-connected layer
        out = self.fc(out)
        
        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data  #HL weight = next(net.parameters()).data.shape
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden
        
n_hidden=512
n_layers=2
net = CharRNN(chars, n_hidden, n_layers)
print(net)
batch_size = 128
seq_length = 100
n_epochs = 20 # start smaller if you are just testing initial behavior
lr=0.001
print_every=10
val_frac=0.1
data = encoded
clip=5

def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network 
    
        Arguments
        ---------
        
        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss
    
    '''
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() #H: the final layer gives linear output
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    if(train_on_gpu):
        net.cuda()
    
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)  # HL h[0] is for hidden states, and h[1] is for cell states, each is [n_layer x n_batch x n_hidden] 
                                         # h[0].requires_grad
        for x, y in get_batches(data, batch_size, seq_length): #pass  x, y = next( get_batches(data, batch_size, seq_length))
            counter += 1
            
            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()  #inputs.requires_grad;targets.requires_grad

            # Creating new variables for the hidden state, otherwise, we'd backprop through the entire training history
            h = tuple([each.data for each in h])    # 可以去掉这一行吗 不行！ h[0].requires_grad  h[0].is_leaf

            # zero accumulated gradients
            net.zero_grad()
            
            # get the output from the model
            output, h = net(inputs, h)             #HL: output.requires_grad   h[0].requires_grad  h[0].is_leaf
            
            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size*seq_length).long()) # self.long() is equivalent to self.to(torch.int64). See to()
            loss.backward()                         #HL: output.requires_grad   h[0].requires_grad
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long()) # 因为LSTM outcome in shape [N,L,H(out)] 而net里设置了 out = out.contiguous().view(-1, self.n_hidden) ; out = self.fc(out)
                
                    val_losses.append(val_loss.item())
                
                net.train() # reset to train mode after iterationg through validation data
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))




# define and print the net
n_hidden=512
n_layers=2

net = CharRNN(chars, n_hidden, n_layers)
print(net)


batch_size = 128
seq_length = 100
n_epochs = 20 # start smaller if you are just testing initial behavior


# train the model
train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)


# change the name, for saving multiple files
model_name = 'rnn_20_epoch.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)


def predict(net, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        
        # tensor inputs
        x = np.array([[net.char2int[char]]])
        x = one_hot_encode(x, len(net.chars))
        inputs = torch.from_numpy(x)
        
        if(train_on_gpu):
            inputs = inputs.cuda()
        
        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = net(inputs, h)

        # get the character probabilities
        p = F.softmax(out, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
        
        # get top characters
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return net.int2char[char], h


def sample(net, size, prime='The', top_k=None):
        
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)



print(sample(net, 1000, prime='Anna', top_k=5))




# Here we have loaded in a model that trained over 20 epochs `rnn_20_epoch.net`
with open('rnn_20_epoch.net', 'rb') as f:
    checkpoint = torch.load(f)
    
loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])


# Sample using a loaded model
print(sample(loaded, 2000, top_k=5, prime="And Levin said"))





































































