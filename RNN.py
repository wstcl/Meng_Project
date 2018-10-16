import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

csvname = 'C:\\Users\\93621\\Desktop\\S_dos.csv'
data = pd.read_csv(csvname)
headers = ['SourceIp','DestIp','SourcePort','destPort','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp']
data.columns = headers
new = [ "RespData" , "WriteData" , "Alarm" ]
for i in new:
    data[i] = np.nan
i = 0
data = data.fillna('0')
packets_num = len(data['Alarm'])
while(i < packets_num):
    if data['funcCode'][i] == 3:
        data['RespData'][i] = data['Register_data'][i]
        if data['SourceIp'][i] == '192.168.100.11':
            if len(data['Register_data'][i+1]) > 10 and len(data['Register_data'][i+3]) > 10:
                data['Alarm'][i:i+3]  = 1
                i = i + 2
                continue
    if data['funcCode'][i] == 16:
        data['WriteData'][i] = data['Register_data'][i]
    if data['Exeption_Code'][i] == 1:
        data['Alarm'][i] = 1
    i = i + 1
del data['Register_data']
data.to_csv('C:\\Users\\93621\\Desktop\\Labeled.csv')

