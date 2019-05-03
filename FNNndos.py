import pandas as pd
import numpy as np
'''def label(csvname,output):
    data = pd.read_csv(csvname)
    headers = ['SourceIp','DestIp','SourcePort','destPort','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time','Alarm']
    data.columns = headers
    X_Label = [i for i in headers if i not in 'Alarm']
    no_packets = data.shape[0]
    data['Alarm']=0
    for i in range(no_packets):
        if data['funcCode'][i]==16:
            if data['Refno'][i]==32210:
                if data['Register_data'][i]<-9 or data['Register_data'][i]>9:
                    data['Alarm'][i]=1
            elif data['Refno'][i]==42210 or data['Refno'][i]==42211:
                if data['Register_data'][i]>95 or data['Register_data'][i]<5:
                    data['Alarm'][i]=1
            elif data['Refno'][i]==42212:
                if data['Register_data'][i] != 95:
                    data['Alarm'][i] = 1
            elif data['Refno'][i]==42213:
                if data['Alarm'][i]!=5:
                    data['Alarm'][i]=1
            elif data['Refno'][i]==42214:
                if data['Register_data'][i]!=80:
                    data['Alarm'][i]=1
            elif data['Refno'][i]==42215:
                if data['Register_data'][i]!=20:
                    data['Alarm'][i]=1
    data.to_csv(output)'''


def label(csvname,output):
    exter_eth = int('000c29d7eaec',16)
    label_index = 21
    data_raw = pd.read_csv(csvname)
    data_raw = np.array(data_raw)
    data = np.zeros((data_raw.shape[0],label_index+1))
    data[:,0:12]=data_raw[:,0:12]
    data[np.argwhere( (data[:,2]==502) & (data[:,6]==16) ),7]=0
    # Reset all labels = 0
    data[:,label_index]=0
    #Function code = 16
    Write_index = np.argwhere(data[:,6]==16)
    Write_index = np.concatenate(Write_index)

    #Refno =32210
    Label = np.argwhere( (data[Write_index,7]==32210) & ( (data[Write_index,8]<-9) | (data[Write_index,8]>9) ))
    data[Write_index[Label],label_index]=1
    #index_32210 = np.argwhere(data[:,7]==32210)
    #data[index_32210,16]=data[index_32210,8]

    # Refno =42210
    Label = np.argwhere((data[Write_index, 7] == 42210) & ((data[Write_index, 8] > 95) | (data[Write_index, 8] < 5) ))
    data[Write_index[Label], label_index] = 1
    #index_42210 = np.argwhere(data[:,7]==42210)
    #data[index_42210,17]=data[index_42210,8]

    # Refno =42211
    Label = np.argwhere((data[Write_index, 7] == 42211) & ((data[Write_index, 8] > 95) | (data[Write_index, 8] < 5)))
    data[Write_index[Label], label_index] = 1
    #index_42211 = np.argwhere(data[:,7]==42211)
    #data[index_42211,18]=data[index_42211,8]

    # Refno =42212
    Label = np.argwhere((data[Write_index, 7] == 42212) & ( data[Write_index, 8] != 95) )
    data[Write_index[Label], label_index] = 1
    #index_42212 = np.argwhere(data[:,7]==42212)
    #data[index_42212,12]=data[index_42212,8]

    # Refno =42213
    Label = np.argwhere((data[Write_index, 7] == 42213) & (data[Write_index, 8] != 5))
    data[Write_index[Label], label_index] = 1
    #index_42213 = np.argwhere(data[:,7]==42213)
    #data[index_42213,13]=data[index_42213,8]

    # Refno =42214
    Label = np.argwhere((data[Write_index, 7] == 42214) & (data[Write_index, 8] != 80))
    data[Write_index[Label], label_index] = 1
    #index_42214 = np.argwhere(data[:,7]==42214)
    #data[index_42214,14]=data[index_42214,8]

    # Refno =42215
    Label = np.argwhere((data[Write_index, 7] == 42215) & (data[Write_index, 8] != 20))
    data[Write_index[Label], label_index] = 1
    #index_42215 = np.argwhere(data[:,7]==42215)
    #data[index_42215,15]=data[index_42215,8]
    
    exter_index = np.argwhere(data[:,19:label_index]==exter_eth)
    mtim_index = np.argwhere(data[exter_index[:,0],0:2]==297913)
    mtim_index = exter_index[mtim_index[:,0],0]
    #mtim_index = np.concatenate(mtim_index)
    data[mtim_index,label_index] = 1
    data = np.delete(data,[19,20],axis=1)
    np.savetxt(output,data,delimiter=',')



def mul_label(csvname,output):
    label_index = 19
    data = pd.read_csv(csvname)
    data = np.array(data)
    data[np.argwhere( (data[:,2]==502) & (data[:,6]==16) ),7]=0
    # Reset all labels = 0
    data[:,label_index]=0
    #Function code = 16
    Write_index = np.argwhere(data[:,6]==16)
    Write_index = np.concatenate(Write_index)

    #Refno =32210
    Label = np.argwhere( (data[Write_index,7]==32210) & ( (data[Write_index,8]<-9) | (data[Write_index,8]>9) ))
    data[Write_index[Label],label_index]=1

    # Refno =42210
    Label = np.argwhere((data[Write_index, 7] == 42210) & ((data[Write_index, 8] > 95) | (data[Write_index, 8] < 5) ))
    data[Write_index[Label], label_index] = 2

    # Refno =42211
    Label = np.argwhere((data[Write_index, 7] == 42211) & ((data[Write_index, 8] > 95) | (data[Write_index, 8] < 5)))
    data[Write_index[Label], label_index] = 3

    # Refno =42212
    Label = np.argwhere((data[Write_index, 7] == 42212) & ( data[Write_index, 8] != 95) )
    data[Write_index[Label], label_index] = 4

    # Refno =42213
    Label = np.argwhere((data[Write_index, 7] == 42213) & (data[Write_index, 8] != 5))
    data[Write_index[Label], label_index] = 5

    # Refno =42214
    Label = np.argwhere((data[Write_index, 7] == 42214) & (data[Write_index, 8] != 80))
    data[Write_index[Label], label_index] = 6

    # Refno =42215
    Label = np.argwhere((data[Write_index, 7] == 42215) & (data[Write_index, 8] != 20))
    data[Write_index[Label], label_index] = 7

    np.savetxt(output,data,delimiter=',')

#label('pcap file/ndos_mitm.csv','pcap file/label_mitm.csv')
mul_label('pcap file/label_AN_3.csv','pcap file/mulabel_AN_3.csv')
    




