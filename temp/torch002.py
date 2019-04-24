import torch
import torch.nn.functional as F     # 激励函数都在这
import torch.utils.data as Data

from torch.autograd import Variable

import random
import pickle
from train_data2 import getdata0
from test_data2 import getdata1




#跑其它測資

#train_data
data7 = getdata0()

data0=[]
data1=[]

i = 0

for row in data7:

    
    data0.append([])

    for o in range(0,520) :
        
        data0[i].append(int(row[o])/100*(-1))

    data1.append([])
    data1[i].append(float(row[520]))
    data1[i].append(float(row[521]))
    data1[i].append(int(row[522]))

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

    for o in range(0,520) :
        
        data0[i].append(int(row[o])/100)

    data1.append([])
    data1[i].append(float(row[520]))
    data1[i].append(float(row[521]))
    data1[i].append(int(row[522]))

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

torch_dataset = Data.TensorDataset(x , y)
#torch_dataset = Data.TensorDataset(data_tensor=x , target_tensor=y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=500,
    shuffle=False,
    num_workers=2,
)

'''
x = torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
'''
#print(y)



net = torch.nn.Sequential(
    torch.nn.Linear(520, 30),
    torch.nn.ReLU(),
    torch.nn.Linear(30, 30),
    torch.nn.ReLU(),
    torch.nn.Linear(30, 30),
    torch.nn.ReLU(),
    torch.nn.Linear(30, 3),
)


print(net)  # net 的结构


optimizer = torch.optim.SGD(net.parameters(), lr=0.03)  # 传入 net 的所有参数, 学习率
#optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)


'''
ppp0 = 0
ppp1 = 0
for t in range(3001):
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值

    #print(prediction)

    loss = loss_func(prediction, y)     # 计算两者的误差

    if ( (t%100) == 0 ) :
        #print(t , loss.data[0])
        print(t)

        #print(prediction)
        #print(y)

        oo = 0
        pp = 0
        
        
        for o in prediction :

            if(oo >1000) :
                break

            #print(o[0].item())
            #print(o[1].item())

            xp1 = float( o[0].item() ) - float( y[oo][0].item() )
            xp2 = float( o[1].item() ) - float( y[oo][1].item() )

            xp33 = ( xp1**2 + xp2**2 ) ** 0.5

            if ( xp33 > 8) :
            #if ( abs(xp1) > 5) or (abs(xp2) > 5) :
                print(t,'  ',oo,'  ',o , '  ' , y[oo] , '  ', xp33 ,  '  <-' )
                pp = pp + 1
            else :
                print(o , '  ' , y[oo] , '  ', xp33 )

            oo = oo + 1
        


        print(pp)
        print(loss.item())
        ppp0 = pp
        ppp1 = loss.item()

        

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
'''



for epoch in range(81):   # 训练所有!整套!数据 3 次
    ppp0 = 0
    ppp1 = 0
    for tt, (batch_x, batch_y) in enumerate(loader):

        b_x = Variable(batch_x).cuda()


        prediction = net(batch_x)     # 喂给 net 训练数据 x, 输出预测值
        #print(prediction)
        loss = loss_func(prediction, batch_y)     # 计算两者的误差

        if ( (epoch%10) == 0 ) :
            #print(t , loss.data[0])
            print(epoch)

            #print(prediction)
            #print(y)

            
            oo = 0
            pp = 0
            
            for o in prediction :
                if(oo >50) :
                    break

                #print(o[0].item())
                #print(o[1].item())

                xp1 = float( o[0].item() ) - float( y[oo][0].item() )
                xp2 = float( o[1].item() ) - float( y[oo][1].item() )

                xp33 = ( xp1**2 + xp2**2 ) ** 0.5

                if ( xp33 > 20) :
                #if ( abs(xp1) > 5) or (abs(xp2) > 5) :
                    print(epoch,'  ',tt,'   ' ,oo,'  ',o , '  ' , y[oo] , '  ', xp33 ,  '  <-' )
                    pp = pp + 1
                else :
                    print(o , '  ' , y[oo] , '  ', xp33 )

                oo = oo + 1
            
            #print(pp)
            #print(loss.item())
            ppp0 = ppp0 + pp
            ppp1 = ppp1 + loss.item()


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
    if(oo >100) :
        break

    xp1 = float( o[0].item() ) - float( test_y[oo][0].item() )
    xp2 = float( o[1].item() ) - float( test_y[oo][1].item() )

    xp33 = ( xp1**2 + xp2**2 ) ** 0.5

    if ( xp33 > 8) :
        #if ( abs(xp1) > 5) or (abs(xp2) > 5) :
        print(oo,'  ',o , '  ' , test_y[oo] , '  ', xp33 ,  '  <-' )
        pp = pp + 1
    else :
        print(o , '  ' , test_y[oo] , '  ', xp33 )

    oo = oo + 1

print(pp)
print(loss.item())

print(ppp0)
print(ppp1)
#print(loss.type)
#print(net.state_dict)

