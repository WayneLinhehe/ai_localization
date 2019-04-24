import csv

# 開啟 CSV 檔案



def getdata0() :

    with open('uj_trainingData.csv', newline='') as csvfile:
    #with open('test.csv', newline='') as csvfile:

        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        data1 = []


        # 以迴圈輸出每一列
        for row in rows:
            #print(row)
            data1.append(row)

        #print('----' , k )

        #print(data1[5000])

        return data1
        

getdata0()
