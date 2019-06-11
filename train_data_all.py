import csv

# 開啟 CSV 檔案



def get_train_data_mid() :

    with open('mid_data\mid_lab_data_train.csv', newline='')  as csvfile:
    
        
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            data1.append( row )
            
        for i in data1 :
            
            i[0] = float( i[0] ) 
            i[1] = float( i[1] ) 
            i[2] = float( i[2] ) 

            i[3] = float( i[3] )
            i[4] = float( i[4] )

            #print(i)
        
        return data1

def get_train_data_mid_half() :

    with open('mid_data\mid_lab_data_train_half.csv', newline='')  as csvfile:
    
        
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            data1.append( row )
            
        for i in data1 :
            
            i[0] = float( i[0] ) 
            i[1] = float( i[1] ) 
            i[2] = float( i[2] ) 

            i[3] = float( i[3] )
            i[4] = float( i[4] )

            #print(i)
        
        return data1

def get_train_data_big() :

    with open('last_data\data_big.csv', newline='')  as csvfile:
    
        
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            data1.append( row )
            
        for i in data1 :
            
            i[0] = float( i[0] ) 
            i[1] = float( i[1] ) 
            i[2] = float( i[2] ) 

            i[3] = float( i[3] )
            i[4] = float( i[4] )

            #print(i)
        
        return data1


def get_train_data_big_half() :

    with open('last_data\data_big_half.csv', newline='')  as csvfile:
    
        
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            data1.append( row )
            
        for i in data1 :
            
            i[0] = float( i[0] ) 
            i[1] = float( i[1] ) 
            i[2] = float( i[2] ) 

            i[3] = float( i[3] )
            i[4] = float( i[4] )

            #print(i)
        
        return data1

def get_train_data_big_quarter() :

    with open('last_data\data_big_quarter.csv', newline='')  as csvfile:
    
        
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            data1.append( row )
            
        for i in data1 :
            
            i[0] = float( i[0] ) 
            i[1] = float( i[1] ) 
            i[2] = float( i[2] ) 

            i[3] = float( i[3] )
            i[4] = float( i[4] )

            #print(i)
        
        return data1

def get_train_data_all() :

    with open('all_data\data_all.csv', newline='')  as csvfile:
    
        
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            data1.append( row )
            
        for i in data1 :
            
            i[0] = float( i[0] ) 
            i[1] = float( i[1] ) 
            i[2] = float( i[2] ) 

            i[3] = float( i[3] )
            i[4] = float( i[4] )

            #print(i)
        
        return data1

def get_train_data_cla() :

    with open('cla_data\cla_data.csv', newline='')  as csvfile:
    
        
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            data1.append( row )
            
        for i in data1 :
            
            i[0] = float( i[0] ) 
            i[1] = float( i[1] ) 
            i[2] = float( i[2] ) 

            i[3] = float( i[3] )

            #print(i)
        
        return data1



def get_train_data_cla_half() :

    with open('cla_data\cla_data.csv', newline='')  as csvfile:
        
        table = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,31,31,32,32,33,33,34,34,35,35,36,36,37,37,38,38,39,39,40,40,41,41,42,42]

        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            data1.append( row )
            
        for i in data1 :
            
            i[0] = float( i[0] ) 
            i[1] = float( i[1] ) 
            i[2] = float( i[2] ) 


            i[3] =  float( table[ int( float( i[3] ) ) -1] )
            #print(i)
        
        return data1

def get_train_data_cla_qua() :

    with open('cla_data\cla_data.csv', newline='')  as csvfile:
        
        table = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20]
        print(len(table))
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            data1.append( row )
            
        for i in data1 :
            
            i[0] = float( i[0] ) 
            i[1] = float( i[1] ) 
            i[2] = float( i[2] ) 


            i[3] =  float( table[ int( float( i[3] ) ) -1] )
            #print(i)
        
        return data1

def trash_use() :

    with open('cla_data\cla_data2.csv', newline='')  as csvfile:
    
        
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 , 0]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            data1.append( row )
            
        for i in data1 :
            
            i[0] = float( i[0] ) 
            i[1] = float( i[1] ) 
            i[2] = float( i[2] ) 
            i[3] = float( i[3] ) 

            i[4] = float( i[4] )

            #print(i)
        
        return data1


#get_train_data_cla_qua()
