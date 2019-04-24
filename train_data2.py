import csv
import math

# 開啟 CSV 檔案



def getdata0() :

    with open('uj_trainingData.csv', newline='') as csvfile:
    #with open('test.csv', newline='') as csvfile:

        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []
    

        # 以迴圈輸出每一列
        for row in rows:
            
            for i in range(0,520) :
                pass
                row[i] = int(row[i])/100*(-1)

            
            row[520] = float(row[520]) * (-1) - 7500
            row[521] = float(row[521]) - 4864850
            row[522] = int(row[522]) #樓層
            row[523] = int(row[523]) #大樓ID

            '''
            if ( row[522] == 2 and row[523] == 0) :
                data1.append(row)
            '''
            data1.append(row)
            
            '''
            for o in range(0,520) :
                
                #print(int(row[o])/100*(-1))

                if (int(row[o]) == 100 or (int(row[o]) < 1 and int(row[o]) > -105) ) :
                    pass
                else  :
                    print(int(row[o]))
            '''


        #print('----' , k )

        #print(data1[5000])

        return data1
        

getdata0()
