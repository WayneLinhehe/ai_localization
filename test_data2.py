import csv

# 開啟 CSV 檔案



def getdata1() :

    with open('uj_validationData.csv', newline='') as csvfile:
    #with open('test.csv', newline='') as csvfile:

        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []


        # 以迴圈輸出每一列
        for row in rows:
            #print(row)
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
        #print('----' , k )

        #print(data1[500])

        return data1

getdata1()
