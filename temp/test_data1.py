import csv

# 開啟 CSV 檔案



def getdata1() :

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
        return data1

getdata1()
