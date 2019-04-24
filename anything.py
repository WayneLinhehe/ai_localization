import csv

# 開啟 CSV 檔案



def getdata0() :

    with open('LAB_DATA_2019_3_10.csv', newline='') as csvfile:
    #with open('test.csv', newline='') as csvfile:

        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []

        data0 = [ 0 , 0 , 0 , 0 , 0 ]
        i = 0
        j = 0
        k = 0

        # 以迴圈輸出每一列
        for row in rows:

            #print(row)

            uu = row[2].split(':')[5]

            if data0[3] != int(row[16]) or data0[4] != int(row[17]) :
                data0[3] = int(row[16])
                data0[4] = int(row[17])
                k = k + 1
                j = 0
            elif uu == 'a709' :
                data0[0] = int(row[13])
            elif uu == 'a56d' :
                data0[1] = int(row[13])
            elif uu == 'a537' :
                data0[2] = int(row[13])

            if data0[0] != 0 and data0[1] != 0 and data0[2] != 0 :
                #print( i , "  " , data0 )
                
                if j != 0 and j < 9 :
                    data1.append( data0 )
                    i = i + 1
                    
                data0 = [ 0 , 0 , 0 , data0[3] , data0[4] ]
                #print( j )
                
                j = j + 1
                
                #data0 = [ 0 , 0 , 0 , 0 , 0 ]

        #print('----' , k )

        
        
        #mean

        data2 = []

        co_x = data1[0][3]
        co_y = data1[0][4]
        da1_sun = [0,0,0,0,0]
        step = 0
        all_step = 0

        for da1_arr in data1:
            all_step = all_step + 1
            
            if co_y != da1_arr[4] :
                

                data_sss = [0,0,0,0,0]

                data_sss[0] = da1_sun[0]/step
                data_sss[1] = da1_sun[1]/step
                data_sss[2] = da1_sun[2]/step
                data_sss[3] = co_x
                data_sss[4] = co_y

                data2.append( data_sss )

                co_x = da1_arr[3]
                co_y = da1_arr[4]
                step = 1
                da1_sun = [da1_arr[0],da1_arr[1],da1_arr[2],0,0]
            else :
                step = step + 1
                da1_sun[0] = da1_sun[0] + da1_arr[0]
                da1_sun[1] = da1_sun[1] + da1_arr[1]
                da1_sun[2] = da1_sun[2] + da1_arr[2]
                
            if all_step == len(data1) :
                data_sss = [0,0,0,0,0]

                data_sss[0] = da1_sun[0]/step
                data_sss[1] = da1_sun[1]/step
                data_sss[2] = da1_sun[2]/step
                data_sss[3] = da1_arr[3]
                data_sss[4] = da1_arr[4]

                data2.append( data_sss )
        """
        print(len(data2))
        """
        for i in data2 :
            
            i[0] = float( i[0] ) * (-1) / 100
            i[1] = float( i[1] ) * (-1) / 100
            i[2] = float( i[2] ) * (-1) / 100

            i[3] = float( i[3] )
            i[4] = float( i[4] )

            print(i)
        
        return data2

getdata0()
