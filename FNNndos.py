import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,accuracy_score
import sys
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


def label(csvname,output='',eva=''):
    exter_eth = int('000c29d7eaec',16)
    label_index = 21
    data_raw = pd.read_csv(csvname)
    data_raw = np.array(data_raw)
    data = np.zeros((data_raw.shape[0],label_index+1))
    data[:,0:12]=data_raw[:,0:12]
    data[:,label_index-2:label_index]=data_raw[:,label_index-2:label_index]
    data[np.argwhere( (data[:,2]==502) & (data[:,6]==16) ),7]=0
    y_pre = data_raw[:,label_index].copy()


    #Function code = 16
    Write_index = np.argwhere(data[:,6]==16)
    Write_index = np.concatenate(Write_index)

    #Refno =32210
    Label = np.argwhere( (data[Write_index,7]==32210) & ( (data[Write_index,8]<-9) | (data[Write_index,8]>9) ))
    data[Write_index[Label],label_index]=1
    index_32210 = np.argwhere(data[:,7]==32210)
    data[index_32210,16]=data[index_32210,8]

    # Refno =42210
    Label = np.argwhere((data[Write_index, 7] == 42210) & ((data[Write_index, 8] > 95) | (data[Write_index, 8] < 5) ))
    data[Write_index[Label], label_index] = 2
    index_42210 = np.argwhere(data[:,7]==42210)
    data[index_42210,17]=data[index_42210,8]

    # Refno =42211
    Label = np.argwhere((data[Write_index, 7] == 42211) & ((data[Write_index, 8] > 95) | (data[Write_index, 8] < 5)))
    data[Write_index[Label], label_index] = 3
    index_42211 = np.argwhere(data[:,7]==42211)
    data[index_42211,18]=data[index_42211,8]

    # Refno =42212
    Label = np.argwhere((data[Write_index, 7] == 42212) & ( data[Write_index, 8] != 95) )
    data[Write_index[Label], label_index] = 4
    index_42212 = np.argwhere(data[:,7]==42212)
    data[index_42212,12]=data[index_42212,8]

    # Refno =42213
    Label = np.argwhere((data[Write_index, 7] == 42213) & (data[Write_index, 8] != 5))
    data[Write_index[Label], label_index] = 5
    index_42213 = np.argwhere(data[:,7]==42213)
    data[index_42213,13]=data[index_42213,8]

    # Refno =42214
    Label = np.argwhere((data[Write_index, 7] == 42214) & (data[Write_index, 8] != 80))
    data[Write_index[Label], label_index] = 6
    index_42214 = np.argwhere(data[:,7]==42214)
    data[index_42214,14]=data[index_42214,8]

    # Refno =42215
    Label = np.argwhere((data[Write_index, 7] == 42215) & (data[Write_index, 8] != 20))
    data[Write_index[Label], label_index] = 7
    index_42215 = np.argwhere(data[:,7]==42215)
    data[index_42215,15]=data[index_42215,8]
    
    exter_index = np.argwhere(data[:,19:label_index]==exter_eth)
    mtim_index = np.argwhere(data[exter_index[:,0],0:2]==297913)
    mtim_index = exter_index[mtim_index[:,0],0]
    #mtim_index = np.concatenate(mtim_index)
    data[mtim_index,label_index] = 8
    data = np.delete(data,[19,20],axis=1)

    start_index = np.argwhere(data[:, 7] == 52210)
    end_index = np.argwhere(data[:, 7] == 52211)
    start_index = np.concatenate(start_index)
    end_index = np.concatenate(end_index)
    Flag_Diff = len(start_index) - len(end_index)
    if Flag_Diff==0:
        for i in range(len(start_index)):
            A_index = np.argwhere(data[start_index[i]:end_index[i],0:2]==5884431)+start_index[i]
            data[A_index[:,0],label_index-2]=10
    elif Flag_Diff == 1:
        for i in range(len(start_index)-1):
            A_index = np.argwhere(data[start_index[i]:end_index[i],0:2]==5884431)+start_index[i]
            data[A_index[:,0],label_index-2]=10
        A_index = np.argwhere(data[start_index[-1]:len(data),0:2]==5884431)+start_index[i]
        data[A_index[:,0],label_index-2]=10    #The reason of label_index-2 is I delete the eth two rows after mitm labeling
    else:
         raise ValueError('Difference between start and end flag is greater than one.')
    flags=np.append(start_index,end_index)
    data = np.delete(data,flags,axis = 0)
    y_pre = np.delete(y_pre,flags,axis=0)
    crc = np.argwhere((data[:, label_index-2] == 10) & (data[:, 9] == 1))
    data[crc, label_index-2] = 9

    print(classification_report(y_pre,data[:,label_index-2]))
    if output!='':
        np.savetxt(output,data,delimiter=',')
    if eva!='':

        pi = precision_score(data[:,label_index-2],y_pre,average='micro')
        pa = precision_score(data[:,label_index-2],y_pre,average='macro')
        ri = recall_score(data[:,label_index-2],y_pre,average='micro')
        ra = recall_score(data[:,label_index-2],y_pre,average='macro')
        fi = f1_score(data[:,label_index-2],y_pre,average='micro')
        fa = f1_score(data[:,label_index-2],y_pre,average='macro')
        acc = accuracy_score(data[:,label_index-2],y_pre)
        print(acc,',',pi,',',ri,',',fi,file=open(eva+"i.csv","a"))
        print(acc,',',pa,',',ra,',',fa,file=open(eva+"a.csv","a"))
    p = precision_score(data[:, label_index - 2], y_pre, average=None)
    r = recall_score(data[:, label_index - 2], y_pre, average=None)
    f = f1_score(data[:, label_index - 2], y_pre, average=None)
    return p, r, f

def dos_mul(csvfile,output):
    data = np.loadtxxt(csvfile,delimiter=',')
    inner = np.argwhere(data[:,0:2]==297913)
    inner = inner[:,0]
    mitm_index = np.argwhere(data[inner,19]==1)
    data[inner[mitm_index],19]=8
    crc = np.argwhere((data[:,19]==1)&(data[:,9]==1))
    data[crc,19]=9
    scan = np.argwhere(data[:,19]==1)
    data[scan,19]=10
    np.savetxt(output,data,delimiter=',')
#label(sys.argv[1],sys.argv[2],sys.argv[3])
#mul_label('pcap file/label_AN_3.csv','pcap file/mulabel_AN_3.csv')
precision = np.zeros((10,11))
recall = np.zeros((10,11))
f1 = np.zeros((10,11))

for i in range(10):
    p,r,f = label('shared/cap/'+str(i)+'.csv')
    precision[i,:]=p
    recall[i,:]=r
    f1[i,:]=f
np.savetxt('shared/evaluation_results/ensemble/precision.csv',precision,delimiter=',')
np.savetxt('shared/evaluation_results/ensemble/recall.csv',recall,delimiter=',')
np.savetxt('shared/evaluation_results/ensemble/f1.csv',f1,delimiter=',')



