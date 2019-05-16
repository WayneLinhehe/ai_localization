import torch
import torch.nn.functional as F     # 激励函数都在这
import torch.utils.data as Data
import math

'''
from train_data1 import get_train_data
from test_data1 import get_test_data
'''

'''
from train_data_big import get_train_data
from train_data_big import get_train_data_half
from train_data_big import get_train_data_quarter
from test_data_big import get_test_data

from train_data1 import get_train_data_small
'''

# new
from train_data_all import get_train_data_mid
from train_data_all import get_train_data_big
from train_data_all import get_train_data_big_half
from train_data_all import get_train_data_big_quarter
from train_data_all import get_train_data_all

from test_data_all import get_test_data_mid
from test_data_all import get_test_data_big
from test_data_all import get_test_data_all



#regresion half data



if __name__ == '__main__':

    #train_sae

    data7 = get_train_data_big()

    data0=[]
    data1=[]

    i = 0

    for row in data7:
        data0.append([])

        for o in range(0,3) :
            if (  math.isnan(row[o] )) :
                data0[i].append(-1)
            else :
                data0[i].append(row[o])

        data1.append([])
        data1[i].append(  row[3])
        data1[i].append(  row[4] )
        #data1[i].append(0)

        i = i + 1


    data0 = torch.FloatTensor(data0 )
    data1 = torch.FloatTensor(data1)

    x = data0

    y = data1



    print('---------------')




    #train_net or sae_net
    
    data7 = get_train_data_big()
    
    data0=[]
    data1=[]

    i = 0

    for row in data7:

        data0.append([])

        for o in range(0,3) :
            if (  math.isnan(row[o] )) :
                data0[i].append(-1)
            else :
                data0[i].append(row[o])

        data1.append([])
        data1[i].append(  row[3])
        data1[i].append(  row[4] )
        #data1[i].append(0)

        i = i + 1

    data0 = torch.FloatTensor(data0 )
    data1 = torch.FloatTensor(data1)

    x_train = data0

    y_train = data1



    print('---------------')


    #test_data
    data8 = get_test_data_big()

    data0=[]
    data1=[]

    i = 0

    for row in data8:
        data0.append([])

        for o in range(0,3) :
            if (  math.isnan(row[o])) :
                data0[i].append(-1)
            else :
                data0[i].append(row[o])
            

        data1.append([])
        data1[i].append(  row[3])
        data1[i].append(  row[4])
        #data1[i].append(0)

        i = i + 1


    data0 = torch.FloatTensor(data0 )

    test_x = data0.cuda()

    data1 = torch.FloatTensor(data1)

    test_y = data1.cuda()



    print('---------------')

    #------ 0.87 -1 0.63 -1 -1 -1 -1 -1 -1
    
    # train
    torch_dataset = Data.TensorDataset(x_train , y_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )


    # SAE
    torch_dataset1 = Data.TensorDataset(x , x)
    loader1 = Data.DataLoader(
        dataset=torch_dataset1,
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )



    class Net_sae(torch.nn.Module):  # 继承 torch 的 Module
        def __init__(self):
            super(Net_sae, self).__init__()     # 继承 __init__ 功能
            # 定义每层用什么样的形式
            self.sae_hidden0 = torch.nn.Linear(3, 3, bias=True)   
            self.sae_hidden1 = torch.nn.Linear(3, 3, bias=True)
            self.sae_hidden2 = torch.nn.Linear(3, 2, bias=True)
            self.sae_hidden3 = torch.nn.Linear(3, 3, bias=True)
            self.sae_hidden4 = torch.nn.Linear(3, 3, bias=True)
            self.sae_predic = torch.nn.Linear(3, 3, bias=True)

        def forward(self, x):   
            x = F.relu(self.sae_hidden0(x))      
            x = F.relu(self.sae_hidden1(x))
            x = F.relu(self.sae_hidden2(x))
            x = F.relu(self.sae_hidden3(x))
            x = F.relu(self.sae_hidden4(x))
            x = self.sae_predic(x)             
            return x

    net_sae = Net_sae()

    #test

    def test_val1() : # NET_SAE test
        prediction = net_sae(test_x)     
        
        all_err = []
                
        for o in prediction :

            err_sum = 0
            for qs in range(0,3) :
                
                err_sum = err_sum + abs( float( o[qs].item() ) - float( test_x[oo][qs].item() ) )
            
            all_err.append(err_sum)

        print(all_err)


    def test_val2() :
        prediction = net_sae_net(test_x)     # 喂给 net 训练数据 x, 输出预测值
        loss2 = loss_func2(prediction, test_y)     # 计算两者的误差

        oo = 0
        pp = 0
        xp44 = 0
                
        for o in prediction :
            
            '''
            print( 'oo :' , oo )
            print(o[0].item())
            print(test_y[oo][0].item())
            print(o[1].item())
            print(test_y[oo][1].item())
            print('+++')
            '''

            xp1 = float( o[0].item() ) - float( test_y[oo][0].item() )
            xp2 = float( o[1].item() ) - float( test_y[oo][1].item() )

            xp33 = ( xp1**2 + xp2**2 ) ** 0.5
            
            xp44 = xp44 + xp33 
            if ( xp33 > 0.5) :
                if ( oo < 100 ) :
                    pass
                pp = pp + 1
            else :
                if ( oo < 100 ) :
                    pass

            oo = oo + 1

        print('epoch : ' , epoch)
        print('loss rate : ' , pp , '   ' , pp / len(prediction) )
        print('loss :' , loss2.item())
        print('mean dis error : ', xp44/ len(prediction) , ' m')
        print()


    net_sae.cuda()

    optimizer1 = torch.optim.Adam(net_sae.parameters(), lr=0.002)  # 传入 net 的所有参数, 学习率
    loss_func1 = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)



    # NET_SAE 

    kk = 0 
    
    for epoch in range(80):   #20
        
        print(kk)

        for (batch_x, batch_y) in loader1:

            b_x = (batch_x).cuda()
            b_y = (batch_y).cuda()


            prediction = net_sae(b_x)
            loss1 = loss_func1(prediction, b_y)    

            optimizer1.zero_grad()   # 清空上一步的残余更新参数值
            loss1.backward()         # 误差反向传播, 计算参数更新值
            optimizer1.step()        # 将参数更新值施加到 net 的 parameters 上
        
        kk = kk + 1

        #test_val1()
    


    net_sae_net = torch.nn.Sequential(*list(net_sae.children())[:-3])


    net_sae_net = torch.nn.Sequential(
        net_sae_net,
        torch.nn.Linear(2, 16, bias=True),
        
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16, bias=True),
        
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2, bias=True),
    )
    

    optimizer2 = torch.optim.Adam(net_sae_net.parameters(), lr=0.001)  # 传入 net 的所有参数, 学习率
    loss_func2 = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

    net_sae_net.cuda()
    
    print(net_sae_net.parameters)


    # freeze
    hj = 0
    for param in net_sae_net.parameters():

        #if ( hj == 0 or hj == 1 or hj == 2 or hj == 3 or hj == 4 or hj == 5 ) :
        #if ( hj == 0 or hj == 1 or hj == 2 ) :
        if ( False ) :
            param.requires_grad = False
        
        print(hj)
        hj = hj + 1

    

    print('******************')



    
    #NET_SAE_NET

    ppp2 = 0
    ppp0 = 0

    for epoch in range(2001):   # NET_SAE_NET
        
        ppp1 = 0
        tt = 0
    
        for (batch_x, batch_y) in loader:

            b_x = (batch_x).cuda()
            b_y = (batch_y).cuda()

            prediction = net_sae_net(b_x)     # 喂给 net 训练数据 x, 输出预测值
            
            loss2 = loss_func2(prediction, b_y)
            
            if ( (epoch%5) == 0 ) :
                
                oo = 0
                pp = 0
                    
                for o in prediction :
                    if(oo >3) :
                        break

                    xp1 = float( o[0].item() ) - float( y[oo][0].item() )
                    xp2 = float( o[1].item() ) - float( y[oo][1].item() )

                    xp33 = ( xp1**2 + xp2**2 ) ** 0.5

                    if ( xp33 > 20) :
                        pp = pp + 1
                    else :
                        pass

                    oo = oo + 1
                
                ppp0 = ppp0 + pp
                ppp1 = ppp1 + loss2.item()
                ppp2 = ppp1

                ppp0 = pp
                
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            tt = tt +1
        
        test_val2()
    





