import torch
import torch.nn.functional as F     # 激励函数都在这
import torch.utils.data as Data
import math

from train_data2 import getdata0
from test_data2 import getdata1



#跑其它測資

#train_data


if __name__ == '__main__':

    data7 = getdata0()

    data0=[]
    data1=[]

    i = 0

    for row in data7:

        
        data0.append([])

        for o in range(0,520) :
            if (  math.isnan(row[o] )) :
                data0[i].append(-1)
            else :
                data0[i].append(row[o])

        data1.append([])
        data1[i].append(  row[520])
        data1[i].append(  row[521] )
        data1[i].append(row[522])
        #data1[i].append(0)

        i = i + 1

    data0 = torch.FloatTensor(data0 )
    data1 = torch.FloatTensor(data1)
    x=data0

    y = data1

    #test_data
    data8 = getdata1()

    data0=[]
    data1=[]

    i = 0

    for row in data8:
        data0.append([])

        for o in range(0,520) :
            if (  math.isnan(row[o])) :
                data0[i].append(-1)
            else :
                data0[i].append(row[o])
            

        data1.append([])
        data1[i].append(  row[520])
        data1[i].append(  row[521])
        data1[i].append(row[522])
        #data1[i].append(0)

        i = i + 1


    data0 = torch.FloatTensor(data0 )

    test_x=data0.cuda()


    data1 = torch.FloatTensor(data1)

    test_y = data1.cuda()

    #------ 0.87 -1 0.63 -1 -1 -1 -1 -1 -1

    torch_dataset = Data.TensorDataset(x , y)
    #torch_dataset = Data.TensorDataset(data_tensor=x , target_tensor=y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )

    torch_dataset1 = Data.TensorDataset(x , x)
    #torch_dataset = Data.TensorDataset(data_tensor=x , target_tensor=y)
    loader1 = Data.DataLoader(
        dataset=torch_dataset1,
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )

    
    
    
    class Net(torch.nn.Module):  # 继承 torch 的 Module
        def __init__(self):
            super(Net, self).__init__()     # 继承 __init__ 功能
            # 定义每层用什么样的形式
            self.hidden0 = torch.nn.Linear(520, 128, bias=False)   
            self.hidden1 = torch.nn.Linear(128, 128, bias=False)
            self.predict = torch.nn.Linear(128, 3, bias=False)   # 输出层线性输出

        def forward(self, x):   # 这同时也是 Module 中的 forward 功能
            
            #x = F.tanh(self.hidden0(x))      # 激励函数(隐藏层的线性值)
            #x = F.tanh(self.hidden1(x))      # 激励函数(隐藏层的线性值)
            x = F.relu(self.hidden0(x))      # 激励函数(隐藏层的线性值)
            x = F.relu(self.hidden1(x))      # 激励函数(隐藏层的线性值)

            x = self.predict(x)             # 输出值
            return x
    
    net = Net()




    class Net_sae(torch.nn.Module):  # 继承 torch 的 Module
        def __init__(self):
            super(Net_sae, self).__init__()     # 继承 __init__ 功能
            # 定义每层用什么样的形式
            self.sae_hidden0 = torch.nn.Linear(520, 256, bias=False)   
            self.sae_hidden1 = torch.nn.Linear(256, 128, bias=False)
            self.sae_hidden2 = torch.nn.Linear(128, 64, bias=False)
            self.sae_hidden3 = torch.nn.Linear(64, 128, bias=False)
            self.sae_hidden4 = torch.nn.Linear(128, 256, bias=False)
            self.sae_predic = torch.nn.Linear(256, 520, bias=False)

        def forward(self, x):   
            x = F.relu(self.sae_hidden0(x))      
            x = F.relu(self.sae_hidden1(x))
            x = F.relu(self.sae_hidden2(x))
            x = F.relu(self.sae_hidden3(x))
            x = F.relu(self.sae_hidden4(x))
            x = self.sae_predic(x)             
            return x

    net_sae = Net_sae()

    
    def weights_init(m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv3d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            #torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            #torch.nn.init.constant_(m.bias.data, 0.0)




    #test

    def test_val0() :
        prediction = net(test_x)     # 喂给 net 训练数据 x, 输出预测值
        loss = loss_func(prediction, test_y)     # 计算两者的误差

        oo = 0
        pp = 0
        xp44 = 0
                
        for o in prediction :
                #print(o[0].item())
                #print(o[1].item())
            

            xp1 = float( o[0].item() ) - float( test_y[oo][0].item() )
            xp2 = float( o[1].item() ) - float( test_y[oo][1].item() )

            xp33 = ( xp1**2 + xp2**2 ) ** 0.5
            xp44 = xp44 + xp33 
            if ( xp33 > 8) :
                if ( oo < 100 ) :
                    pass
                pp = pp + 1
            else :
                if ( oo < 100 ) :
                    pass

            oo = oo + 1

        print('epoch : ' , epoch)
        print('loss rate : ' , pp , '   ' , pp/1111)
        print('loss :' , loss.item())
        print('mean dis error : ', xp44/1111 , ' m')
        print()

        #print(ppp0)
        #print(ppp2)
        #print(loss.type)
        #print(net.state_dict)


    def test_val1() : # NET_SAE test
        prediction = net_sae(test_x)     
        
        all_err = []
                
        for o in prediction :
                #print(o[0].item())
                #print(o[1].item())

            err_sum = 0
            for qs in range(0,520) :
                
                err_sum = err_sum + abs( float( o[qs].item() ) - float( test_x[oo][qs].item() ) )
            
            all_err.append(err_sum)

        print(all_err)

        #print(ppp0)
        #print(ppp2)
        #print(loss.type)
        #print(net.state_dict)


    def test_val2() :
        prediction = net_sae_net(test_x)     # 喂给 net 训练数据 x, 输出预测值
        loss2 = loss_func2(prediction, test_y)     # 计算两者的误差

        oo = 0
        pp = 0
        xp44 = 0
                
        for o in prediction :
                #print(o[0].item())
                #print(o[1].item())
            

            xp1 = float( o[0].item() ) - float( test_y[oo][0].item() )
            xp2 = float( o[1].item() ) - float( test_y[oo][1].item() )

            xp33 = ( xp1**2 + xp2**2 ) ** 0.5
            xp44 = xp44 + xp33 
            if ( xp33 > 8) :
                if ( oo < 100 ) :
                    pass
                pp = pp + 1
            else :
                if ( oo < 100 ) :
                    pass

            oo = oo + 1

        print('epoch : ' , epoch)
        print('loss rate : ' , pp , '   ' , pp/1111)
        print('loss :' , loss2.item())
        print('mean dis error : ', xp44/1111 , ' m')
        print()

        #print(ppp0)
        #print(ppp2)
        #print(loss.type)
        #print(net.state_dict)





    #print(net)  # net 的结构

    #optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)  # 传入 net 的所有参数, 学习率
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

    net.apply(weights_init)
    net.cuda()

    net_sae.cuda()

    optimizer1 = torch.optim.Adam(net_sae.parameters(), lr=0.001)  # 传入 net 的所有参数, 学习率
    loss_func1 = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)



    
    for epoch in range(20):   # NET_SAE
        
        kk = 0 
        ooo = 0
        for (batch_x, batch_y) in loader1:

            b_x = (batch_x).cuda()
            b_y = (batch_y).cuda()


            prediction = net_sae(b_x)
            #print(prediction[0])


            loss1 = loss_func1(prediction, b_y)    

            
            if ( (epoch%3) == 0 and ooo < 20) :
                
                #print(epoch)
                    
                oo = 0
                    
                for o in prediction :
                    if(oo >3) :
                        break

                    err_sum = 0
                    for qs in range(0,520) :
                        err_sum = err_sum + abs( float( o[qs].item() ) - float( test_x[oo][qs].item() ) )
                    
                    print(epoch , '  ' , kk , '  ' , err_sum)
                    oo = oo + 1

                ooo = ooo + 1


            kk = kk + 1
                


            optimizer1.zero_grad()   # 清空上一步的残余更新参数值
            loss1.backward()         # 误差反向传播, 计算参数更新值
            optimizer1.step()        # 将参数更新值施加到 net 的 parameters 上
            

        #test_val1()
    





    '''
    print('------')
    print(net_sae)
    print('------')

    print(len(list(net_sae.modules()) ))# to inspect the modules of your model
    '''


    net_sae_net = torch.nn.Sequential(*list(net_sae.children())[:-3])

    '''
    print('------')
    print(net_sae_net)
    print('------')
    '''

    net_sae_net = torch.nn.Sequential(
        net_sae_net,
        torch.nn.Linear(64, 128, bias=False),
        
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128, bias=False),
        
        torch.nn.ReLU(),
        torch.nn.Linear(128, 3, bias=False),
    )
    

    optimizer2 = torch.optim.Adam(net_sae_net.parameters(), lr=0.001)  # 传入 net 的所有参数, 学习率
    loss_func2 = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

    #net_sae_net.apply(weights_init)
    net_sae_net.cuda()
    
    '''
    print('------')
    print(net_sae_net)
    print('------')
    for param in net_sae_net.parameters():
        print('+++++++')
        print(len(param))
        #print(param.data)
    '''

    print(net_sae_net.parameters)


    hj = 0
    for param in net_sae_net.parameters():

        #if ( hj == 0 or hj == 1 or hj == 2 or hj == 3 or hj == 4 or hj == 5 ) :
        if ( hj == 0 or hj == 1 or hj == 2 ) :
            param.requires_grad = False

        hj = hj + 1

    

    #NET
    '''
    ppp2 = 0
    ppp0 = 0

    for epoch in range(21):   # NET
        
        ppp1 = 0
        tt = 0
    
        for (batch_x, batch_y) in loader:

            b_x = (batch_x).cuda()
            b_y = (batch_y).cuda()


            prediction = net(b_x)     # 喂给 net 训练数据 x, 输出预测值
            
            loss = loss_func(prediction, b_y)
            


            if ( (epoch%5) == 0 ) :
                
                #print(epoch)
                    
                oo = 0
                pp = 0
                    
                for o in prediction :
                    if(oo >3) :
                        break

                    xp1 = float( o[0].item() ) - float( y[oo][0].item() )
                    xp2 = float( o[1].item() ) - float( y[oo][1].item() )

                    xp33 = ( xp1**2 + xp2**2 ) ** 0.5

                    if ( xp33 > 20) :
                    #if ( abs(xp1) > 5) or (abs(xp2) > 5) :
                        #print(epoch,'  ',tt,'   ' ,oo,'  ',o , '  ' , y[oo] , '  ', xp33 ,  '  <--' )
                        pass
                        pp = pp + 1
                    else :
                        #print(epoch,'  ',tt,'   ' ,oo,'  ',o , '  ' , y[oo] , '  ', xp33)
                        pass

                    oo = oo + 1
                    
                
                ppp0 = ppp0 + pp
                ppp1 = ppp1 + loss.item()
                ppp2 = ppp1

                #print(pp)
                #print(loss.item())
                #print(ppp1)
                ppp0 = pp
                


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tt = tt +1

        test_val0()
    '''



    
    #NET_SAE_NET

    ppp2 = 0
    ppp0 = 0

    for epoch in range(21):   # NET_SAE_NET
        
        ppp1 = 0
        tt = 0
    
        for (batch_x, batch_y) in loader:

            b_x = (batch_x).cuda()
            b_y = (batch_y).cuda()


            prediction = net_sae_net(b_x)     # 喂给 net 训练数据 x, 输出预测值
            
            loss2 = loss_func2(prediction, b_y)
            


            if ( (epoch%5) == 0 ) :
                
                #print(epoch)
                    
                oo = 0
                pp = 0
                    
                for o in prediction :
                    if(oo >3) :
                        break

                    xp1 = float( o[0].item() ) - float( y[oo][0].item() )
                    xp2 = float( o[1].item() ) - float( y[oo][1].item() )

                    xp33 = ( xp1**2 + xp2**2 ) ** 0.5

                    if ( xp33 > 20) :
                    #if ( abs(xp1) > 5) or (abs(xp2) > 5) :
                        #print(epoch,'  ',tt,'   ' ,oo,'  ',o , '  ' , y[oo] , '  ', xp33 ,  '  <--' )
                        pass
                        pp = pp + 1
                    else :
                        #print(epoch,'  ',tt,'   ' ,oo,'  ',o , '  ' , y[oo] , '  ', xp33)
                        pass

                    oo = oo + 1
                    
                
                ppp0 = ppp0 + pp
                ppp1 = ppp1 + loss2.item()
                ppp2 = ppp1

                #print(pp)
                #print(loss.item())
                #print(ppp1)
                ppp0 = pp
                


            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            tt = tt +1

        test_val2()
    
    





