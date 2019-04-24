import torch
import torch.nn.functional as F     # 激励函数都在这

from torch.autograd import Variable

import random
import pickle
from train_data1 import getdata0
from test_data1 import getdata1






#train_data
data7 = getdata0()

data0=[]
data1=[]

i = 0

for row in data7:
    data0.append([])
    data0[i].append(row[0]/100)
    data0[i].append(row[1]/100)
    data0[i].append(row[2]/100)

    data1.append([])
    data1[i].append(row[3])
    data1[i].append(row[4])

    i = i + 1

data0 = torch.FloatTensor(data0 )

x=data0

data1 = torch.FloatTensor(data1)

y = data1




#test_data
data8 = getdata1()

data0=[]
data1=[]

i = 0

for row in data8:
    data0.append([])
    data0[i].append(row[0]/100)
    data0[i].append(row[1]/100)
    data0[i].append(row[2]/100)

    data1.append([])
    data1[i].append(row[3])
    data1[i].append(row[4])

    i = i + 1

#print('--------------' , data0)
data0 = torch.FloatTensor(data0 )

test_x=data0
#print('+++++' , x)

#
#x = torch.unsqueeze(x, dim=0) 
#print('*****' , x)

#print('--------------' , data1)
data1 = torch.FloatTensor(data1)

test_y = data1

#


'''
x = torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
'''
#print(y)


class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden1 , n_hidden2 , n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden0 = torch.nn.Linear(n_feature, n_hidden1)   # 隐藏层线性输出

        self.hidden1 = torch.nn.Linear(n_hidden1, n_hidden2)

        self.predict = torch.nn.Linear(n_hidden2, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden0(x))      # 激励函数(隐藏层的线性值)
        x = F.relu(self.hidden1(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x

net = Net(n_feature=3, n_hidden1=10,n_hidden2=10, n_output=2)

print(net)  # net 的结构


optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

ppp0 = 0
ppp1 = 0
for t in range(30001):
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值

    #print(prediction)

    loss = loss_func(prediction, y)     # 计算两者的误差

    if ( (t%5000) == 0 ) :
        #print(t , loss.data[0])
        print(t)

        #print(prediction)
        #print(y)

        oo = 0
        pp = 0
        
        
        for o in prediction :
            #print(o[0].item())
            #print(o[1].item())

            xp1 = float( o[0].item() ) - float( y[oo][0].item() )
            xp2 = float( o[1].item() ) - float( y[oo][1].item() )

            xp33 = ( xp1**2 + xp2**2 ) ** 0.5

            if ( xp33 > 6.13) :
            #if ( abs(xp1) > 5) or (abs(xp2) > 5) :
                print(o , '  ' , y[oo] , ' <-' )
                pp = pp + 1
            else :
                print(o , '  ' , y[oo] )

            oo = oo + 1

        print(pp)
        print(loss.data[0])
        ppp0 = pp
        ppp1 = loss.data[0]

        

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上





#test
prediction = net(test_x)     # 喂给 net 训练数据 x, 输出预测值
loss = loss_func(prediction, test_y)     # 计算两者的误差

oo = 0
pp = 0
        
for o in prediction :
            #print(o[0].item())
            #print(o[1].item())

    xp1 = float( o[0].item() ) - float( test_y[oo][0].item() )
    xp2 = float( o[1].item() ) - float( test_y[oo][1].item() )

    xp33 = ( xp1**2 + xp2**2 ) ** 0.5

    if ( xp33 > 6.13) :
    #if ( abs(xp1) > 5) or (abs(xp2) > 5) :
        print(o , '  ' , test_y[oo] , ' <-' )
        pp = pp + 1
    else :
        print(o , '  ' , test_y[oo] )

    oo = oo + 1

print(pp)
print(loss.data[0])

print(ppp0)
print(ppp1)
#print(loss.type)
#print(net.state_dict)

