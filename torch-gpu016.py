import torch
import torch.nn.functional as F     # 激励函数都在这
import torch.utils.data as Data
import math

'''
from train_data_cla import get_train_data
from test_data_cla import get_test_data
'''
from train_data_cla1 import get_train_data
from test_data_cla1 import get_test_data




#classification


if __name__ == '__main__':

    data7 = get_train_data()

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
        '''
        for po in range(3,51) :
            data1[i].append(  row[po])
        '''
        #data1[i].append(0)

        i = i + 1

    data0 = torch.FloatTensor(data0 )
    data1 = torch.LongTensor(data1)
    x=data0

    y = data1

    #test_data
    data8 = get_test_data()

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
        '''
        for po in range(3,51) :
            data1[i].append(  row[po])
        '''
        #data1[i].append(0)

        i = i + 1


    data0 = torch.FloatTensor(data0 )

    test_x=data0.cuda()


    data1 = torch.LongTensor(data1)

    test_y = data1.cuda()

    #------ 0.87 -1 0.63 -1 -1 -1 -1 -1 -1

    torch_dataset = Data.TensorDataset(x , y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )

    torch_dataset1 = Data.TensorDataset(x , x)
    loader1 = Data.DataLoader(
        dataset=torch_dataset1,
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )

    

    def test_val2() :
        sum = 0

        prediction = net_sae_net(test_x)     # 喂给 net 训练数据 x, 输出预测值
        loss2 = loss_func2(prediction,torch.max(test_y, 1)[0])

        for po in range( 0 , len( torch.argmax(prediction, 1) ) ):
            

            xp1 = torch.argmax(prediction, 1)[po].item()
            #print('xp1 -> ' , xp1)
            xp2 = torch.max(test_y, 1)[0][po].item()
            #print('xp2 -> ' , xp2)
            if ( xp1 == xp2 ) :
                sum = sum + 1
        
        sum = sum / len( torch.argmax(prediction, 1) )
        
        print('  val_acc = ' , sum)


    net_sae_net = torch.nn.Sequential(
        
        torch.nn.Linear(3, 256, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 48, bias=False),
        torch.nn.Softmax(), 
    )
    

    optimizer2 = torch.optim.Adam(net_sae_net.parameters(), lr=0.001)  # 传入 net 的所有参数, 学习率
    loss_func2 = torch.nn.CrossEntropyLoss()      # 预测值和真实值的误差计算公式 (均方差)

    net_sae_net.cuda()
    
    print(net_sae_net.parameters)

    
    
    #NET_SAE_NET
    
    for epoch in range(2000):   # NET_SAE_NET
        
        tt = 0
        sum = 0
    
        for (batch_x, batch_y) in loader:

            

            b_x = (batch_x).cuda()
            b_y = (batch_y).cuda()

            prediction = net_sae_net(b_x)     # 喂给 net 训练数据 x, 输出预测值

            #print(torch.argmax(prediction, 1))
            #print(torch.max(b_y, 1)[0])
            
            
            loss2 = loss_func2(prediction,torch.max(b_y, 1)[0])

            po1 = 0
            for po in range( 0 , len( torch.argmax(prediction, 1) ) ):
                

                xp1 = torch.argmax(prediction, 1)[po].item()
                #print('xp1 -> ' , xp1)
                xp2 = torch.max(b_y, 1)[0][po].item()
                #print('xp2 -> ' , xp2)
                if ( xp1 == xp2 ) :
                    po1 = po1 + 1 
            po1 = po1 / 10
            sum = sum + po1
            #print(po1)

            #print(loss2)
            

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            tt = tt +1

        sum = sum / tt

        print( epoch , '  acc = ' , sum)
        test_val2()
    
    





