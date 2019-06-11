import torch
import torch.nn.functional as F     # 激励函数都在这
import torch.utils.data as Data
import math
import random

import os


'''
from train_data_cla import get_train_data
from test_data_cla import get_test_data
'''
from train_data_cla1 import get_train_data
from test_data_cla1 import get_test_data

from train_data_all import trash_use
from train_data_all import get_train_data_cla



#classification


if __name__ == '__main__':
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    batch_num = 10

    for rihibgfd in range(0, 1) :
    
        min_all = 0
        step = 0

        data6 = trash_use()
        #data6 = get_train_data_cla()

        data0=[] #for train
        data1=[] #for test

        vv = data6[0][4]
        step_left = 0

        for step_right in range( 0, len(data6) ) :

            if ( ( step_right != len(data6)-1 ) and vv == data6[step_right][4] ) :
                pass
            else :
                kc = (step_right-1)
                ran = random.randint( step_left , kc )
                
                for i in range( step_left, step_right ) :
                    data6[i][4] = data6[i][4]-1
                    if ( ran == i ) :
                        data1.append( data6[i] )
                    else :
                        data0.append( data6[i] )
                
                vv = data6[step_right][4]
                step_left = step_right

        print(data0, '\n')
        print(data1, '\n')

        train_data_done = data0
        test_data_done = data1




        #


        #data7 = get_train_data()
        data7 = train_data_done

        data0=[]
        data1=[]

        i = 0

        for row in data7:

            
            data0.append([])

            for o in range(0,4) :
                if (  math.isnan(row[o] )) :
                    data0[i].append(-1)
                else :
                    data0[i].append(row[o])

            data1.append([])
            
            data1[i].append(  float(row[4]))
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
        #data8 = get_test_data()
        data8 = test_data_done

        data0=[]
        data1=[]

        i = 0

        for row in data8:
            data0.append([])

            for o in range(0,4) :
                if (  math.isnan(row[o])) :
                    data0[i].append(-1)
                else :
                    data0[i].append(row[o])
                

            data1.append([])
            data1[i].append(  float(row[4]))
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
            batch_size=batch_num,
            shuffle=True,
            num_workers=1,
        )

        

        def test_val2() :
            sum = 0

            global min_all
            global step

            prediction = net_sae_net(test_x)     # 喂给 net 训练数据 x, 输出预测值
            loss2 = loss_func2(prediction,torch.max(test_y, 1)[0])

            for po in range( 0 , len( torch.argmax(prediction, 1) ) ):
                

                xp1 = torch.argmax(prediction, 1)[po].item()
                #print()
                xp2 = torch.max(test_y, 1)[0][po].item()
                #print( 'xp1 -> ' , xp1 , '    xp2 -> ' , xp2)
                if ( xp1 == xp2 ) :
                    sum = sum + 1
            
            sum = sum / len( torch.argmax(prediction, 1) )

            if ( min_all < sum ) :
                min_all = sum
                step = epoch
            
            print(' epoch :   ' , epoch , '    val_acc = ' , sum , ' max_val_acc = ' , min_all) 

        '''
        net_sae_net = torch.nn.Sequential(
            
            
            torch.nn.Linear(4, 1024, bias=True),
            torch.nn.BatchNorm1d(1024 , affine=True),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024, bias=True),
            torch.nn.BatchNorm1d(1024 , affine=True),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 90, bias=True),
            torch.nn.Softmax(dim=1), 
        )
        '''

        class Net_sae_net(torch.nn.Module):  # 继承 torch 的 Module
            def __init__(self):
                super(Net_sae_net, self).__init__()     # 继承 __init__ 功能
                # 定义每层用什么样的形式
                self.hidden0 = torch.nn.Linear(4, 1024, bias=True)

                self.bn0 = torch.nn.BatchNorm1d(1024, affine=True)

                self.hidden1 = torch.nn.Linear(1024, 1024, bias=True)

                self.bn1 = torch.nn.BatchNorm1d(1024, affine=True)

                self.predict = torch.nn.Linear(1024, 90)   # 输出层线性输出
                self.soft = torch.nn.Softmax()

            def forward(self, x):   # 这同时也是 Module 中的 forward 功能
                
                

                x = self.hidden0(x)
                #print(x.size()[0])
                #if (x.size()[0] > 1) :
                x = self.bn0(x)

                x = F.relu(x)

                x = self.hidden1(x)

                #if (x.size()[0] > 1) :
                x = self.bn1(x)

                x = F.relu(x)

                x = self.predict(x)             # 输出值
                x = self.soft(x)
                return x
        
        net_sae_net = Net_sae_net()
        

        optimizer2 = torch.optim.Adam(net_sae_net.parameters(), lr=0.0002)  # 传入 net 的所有参数, 学习率
        loss_func2 = torch.nn.CrossEntropyLoss()      # 预测值和真实值的误差计算公式 (均方差)

        net_sae_net.cuda()
        
        print(net_sae_net.parameters)

        
        
        #NET_SAE_NET
        
        for epoch in range(500):   # NET_SAE_NET
            
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

                '''
                print('+++')
                print(torch.argmax(prediction, 1))
                print('---')
                '''

                for po in range( 0 , len( torch.argmax(prediction, 1) ) ):
                    

                    xp1 = torch.argmax(prediction, 1)[po].item()
                    #print('xp1 -> ' , xp1)
                    xp2 = torch.max(b_y, 1)[0][po].item()
                    #print('xp2 -> ' , xp2)
                    if ( xp1 == xp2 ) :
                        po1 = po1 + 1 
                po1 = po1 / batch_num
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
        
        torch.save(net_sae_net, 'netdolo.pkl')

        fp = open('text\\trash.txt', 'a')
    
        # 寫入 This is a testing! 到檔案
        fp.write( ' step : ' + str(step) + '  val : ' + str(min_all) + '\n')
        
        # 關閉檔案
        fp.close()