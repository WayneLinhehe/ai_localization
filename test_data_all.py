import csv

# 開啟 CSV 檔案



def get_test_data_mid() :

    with open('mid_data\mid_lab_data_test.csv', newline='')  as csvfile:
    
        
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

def get_test_data_mid_half() :

    with open('mid_data\mid_lab_data_test_half.csv', newline='')  as csvfile:
    
        
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


def get_test_data_big() :

    with open('last_data\data_big_test.csv', newline='')  as csvfile:
    
        
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

def get_test_data_big_half() :

    with open('last_data\data_big_test_half.csv', newline='')  as csvfile:
    
        
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



def get_test_data_all() :

    with open('all_data\data_all_test.csv', newline='')  as csvfile:
    
        
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

#get_test_data_mid()
