import torch
import torch.nn.functional as F     # 激励函数都在这
import torch.utils.data as Data
import math


# new
from train_data_all import get_train_data_mid
from train_data_all import get_train_data_big
from train_data_all import get_train_data_big_half
from train_data_all import get_train_data_big_quarter
from train_data_all import get_train_data_all

from test_data_all import get_test_data_mid
from test_data_all import get_test_data_big
from test_data_all import get_test_data_all



#

# mid hide 6

if __name__ == '__main__':

    all_zero = 0

    print('---------------')

    #train_net 
    
    data7 = get_train_data_mid()
    
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
    data8 = get_test_data_mid()

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


    
    class Net(torch.nn.Module):  # 继承 torch 的 Module
        def __init__(self):
            super(Net, self).__init__()     # 继承 __init__ 功能
            # 定义每层用什么样的形式
            self.hidden0 = torch.nn.Linear(3, 128, bias=True)   
            self.hidden1 = torch.nn.Linear(128, 128, bias=True)
            self.hidden2 = torch.nn.Linear(128, 128, bias=True)
            self.hidden3 = torch.nn.Linear(128, 128, bias=True)
            self.hidden4 = torch.nn.Linear(128, 128, bias=True)
            self.hidden5 = torch.nn.Linear(128, 128, bias=True)
            self.predict = torch.nn.Linear(128, 2, bias=True)   # 输出层线性输出

        def forward(self, x):   # 这同时也是 Module 中的 forward 功能
            '''
            x = F.relu(self.hidden0(x))      
            x = F.relu(self.hidden1(x))      
            x = F.relu(self.hidden2(x))
            '''
            x = F.selu(self.hidden0(x))      
            x = F.selu(self.hidden1(x))      
            x = F.selu(self.hidden2(x))
            x = F.selu(self.hidden3(x))
            x = F.selu(self.hidden4(x))
            x = F.selu(self.hidden5(x))

            x = self.predict(x)            
            return x
    
    net = Net()


    #test

    def test_val0() :

        global all_zero

        prediction = net(test_x)     # 喂给 net 训练数据 x, 输出预测值
        loss = loss_func(prediction, test_y)     # 计算两者的误差

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

        print( 'epoch : ' , epoch , ' ,  mean dis error : ', xp44 / len(prediction) , ' m')

        
        if ( (epoch % 100) == 0 ) :
            all_zero = 0
        else :
            all_zero = all_zero + ( xp44 / len(prediction) )
            print( ' mean mean dis error : ',( all_zero / (epoch % 100) ) , ' m')

    #optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)  # 传入 net 的所有参数, 学习率
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

    net.cuda()


    print(net)

    #NET

    
    ppp2 = 0
    ppp0 = 0

    for epoch in range(2001):   # NET
        
        ppp1 = 0
        tt = 0
    
        for (batch_x, batch_y) in loader:

            b_x = (batch_x).cuda()
            b_y = (batch_y).cuda()

            prediction = net(b_x)   
            
            loss = loss_func(prediction, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tt = tt +1
        
        test_val0()

    print('+++++++++++')

