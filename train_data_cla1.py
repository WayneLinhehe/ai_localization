import csv

# 開啟 CSV 檔案



def get_train_data() :

    with open('LAB_DATA_2019_5_15.csv', newline='') as csvfile:
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

            if data0[3] != int(row[16])  :
                data0[3] = int(row[16])
                
                k = k + 1
                j = 0
            elif uu == 'a5dd' :              #後方遠離門角
                data0[0] = int(row[13])
            elif uu == 'a5ab' :              #後方門左上角
                data0[1] = int(row[13])
            elif uu == 'a6dd' :              #前方門左上角
                data0[2] = int(row[13])

            if data0[0] != 0 and data0[1] != 0 and data0[2] != 0 :
                #print( i , "  " , data0 )
                
                if j != 0 and j < 5 : #9
                    data1.append( data0 )
                    i = i + 1
                    
                data0 = [ 0 , 0 , 0 , data0[3] ]
                #print( j )
                
                j = j + 1
                
                #data0 = [ 0 , 0 , 0 , 0 , 0 ]

        #print('----' , k )

        for i in data1 :
            
            i[0] = float( i[0] ) * (-1) / 100
            i[1] = float( i[1] ) * (-1) / 100
            i[2] = float( i[2] ) * (-1) / 100

            i[3] = float( i[3] )

            print(i)
        '''
        for j in data1 :
            pass
            if ( j[3] > 1.9 and j[3] < 2.1 ) :
                j[3] = ( j[4] - 3 ) / 4

            elif  ( j[3] > 16.9 and j[3] < 17.1 ) :
                j[3] = ( j[4] - 3 ) / 4 + 16

            elif  ( j[3] > 25 ) :
                j[3] = ( j[4] - 3 ) / 4 + 32
                
            j.pop(4)

            print(j)
        '''
        '''
        for k in data1 :
            #print(k)
            
            pq = k[3]

            k.pop(3)
            
            for r in range(0,48) :
                k.append(0)
            k[ int(pq) + 3 ] = 1
            #print(k)
        '''
        print('hey')
        print('hey')
        print('hey')
        print('star')
        print('dash?')

        return data1

if __name__ == '__main__':
    get_train_data()
