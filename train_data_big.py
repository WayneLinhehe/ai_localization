import csv

# 開啟 CSV 檔案



def get_train_data() :

    #with open('LAB_DATA_ALL_2.csv', newline='')  as csvfile:
    with open('LAB_DATA_ALL_2.csv', newline='')  as csvfile:
    
        
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            #if ( (( int(row[17]) -3 ) %8 ) == 0   ) :
            if (  False  ) :
                pass
            else :

                uu = row[2].split(':')[5]

                if data0[3] != int(row[16]) or data0[4] != int(row[17]) :
                    data0[3] = int(row[16])
                    data0[4] = int(row[17])
                    k = k + 1
                    j = 0
                elif uu == 'a5ab' :
                #elif uu == 'a709' :
                    data0[0] = int(row[13])
                elif uu == 'a5dd' :
                #elif uu == 'a56d' :
                    data0[1] = int(row[13])
                elif uu == 'a6dd' :
                #elif uu == 'a537' :
                    data0[2] = int(row[13])

                
                if data0[0] != 0 and data0[1] != 0 and data0[2] != 0 :
                    #print( i , "  " , data0 )
                    
                    if j != 0 and j < 40 :
                    #if j != 0 and j < 9 :
                        data1.append( data0 )
                        i = i + 1
                        
                    data0 = [ 0 , 0 , 0 , data0[3] , data0[4] ]
                    #print( j )
                    
                    j = j + 1
                    
                #data0 = [ 0 , 0 , 0 , 0 , 0 ]

        #print('----' , k )

        for i in data1 :
            
            i[0] = float( i[0] ) * (-1) / 100
            i[1] = float( i[1] ) * (-1) / 100
            i[2] = float( i[2] ) * (-1) / 100

            i[3] = float( i[3] )
            i[4] = float( i[4] )

            #print(i)
        
        return data1


def get_train_data_half() :

    with open('LAB_DATA_ALL_2_half.csv', newline='')  as csvfile:
    
        
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            #if ( (( int(row[17]) -3 ) %8 ) == 0   ) :
            if (  False  ) :
                pass
            else :

                uu = row[2].split(':')[5]

                if data0[3] != int(row[16]) or data0[4] != int(row[17]) :
                    data0[3] = int(row[16])
                    data0[4] = int(row[17])
                    k = k + 1
                    j = 0
                elif uu == 'a5ab' :
                #elif uu == 'a709' :
                    data0[0] = int(row[13])
                elif uu == 'a5dd' :
                #elif uu == 'a56d' :
                    data0[1] = int(row[13])
                elif uu == 'a6dd' :
                #elif uu == 'a537' :
                    data0[2] = int(row[13])

                
                if data0[0] != 0 and data0[1] != 0 and data0[2] != 0 :
                    #print( i , "  " , data0 )
                    
                    if j != 0 and j < 10 :
                    #if j != 0 and j < 9 :
                        data1.append( data0 )
                        i = i + 1
                        
                    data0 = [ 0 , 0 , 0 , data0[3] , data0[4] ]
                    #print( j )
                    
                    j = j + 1
                    
                #data0 = [ 0 , 0 , 0 , 0 , 0 ]

        #print('----' , k )

        for i in data1 :
            
            i[0] = float( i[0] ) * (-1) / 100
            i[1] = float( i[1] ) * (-1) / 100
            i[2] = float( i[2] ) * (-1) / 100

            i[3] = float( i[3] )
            i[4] = float( i[4] )

            #print(i)
        
        return data1


def get_train_data_quarter() :

    with open('LAB_DATA_ALL_2_quarter.csv', newline='')  as csvfile:
    
        
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            #if ( (( int(row[17]) -3 ) %8 ) == 0   ) :
            if (  False  ) :
                pass
            else :

                uu = row[2].split(':')[5]

                if data0[3] != int(row[16]) or data0[4] != int(row[17]) :
                    data0[3] = int(row[16])
                    data0[4] = int(row[17])
                    k = k + 1
                    j = 0
                elif uu == 'a5ab' :
                #elif uu == 'a709' :
                    data0[0] = int(row[13])
                elif uu == 'a5dd' :
                #elif uu == 'a56d' :
                    data0[1] = int(row[13])
                elif uu == 'a6dd' :
                #elif uu == 'a537' :
                    data0[2] = int(row[13])

                
                if data0[0] != 0 and data0[1] != 0 and data0[2] != 0 :
                    #print( i , "  " , data0 )
                    
                    if j != 0 and j < 10 :
                    #if j != 0 and j < 9 :
                        data1.append( data0 )
                        i = i + 1
                        
                    data0 = [ 0 , 0 , 0 , data0[3] , data0[4] ]
                    #print( j )
                    
                    j = j + 1
                    
                #data0 = [ 0 , 0 , 0 , 0 , 0 ]

        #print('----' , k )

        for i in data1 :
            
            i[0] = float( i[0] ) * (-1) / 100
            i[1] = float( i[1] ) * (-1) / 100
            i[2] = float( i[2] ) * (-1) / 100

            i[3] = float( i[3] )
            i[4] = float( i[4] )

            #print(i)
        
        return data1


#get_train_data_half()
