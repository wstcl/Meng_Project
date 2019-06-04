import csv
import statistics
from sklearn.metrics import classification_report
from keras.models import load_model
import numpy as np
import os
import sys
def process(csvname,output):
    with open(csvname) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        F1 = []
        Pre = []
        rec = []
        acc = []
        loss = []

        F1_mean = []
        Pre_mean = []
        rec_mean = []
        acc_mean = []
        loss_mean = []
        F1_std = []
        Pre_std = []
        rec_std = []
        acc_std = []
        loss_std = []

        for row in readCSV:
            if len(row)!=1:
                #try:
                #    row[2]
                #except IndexError:
                #    print('error',row)
                print(row)
                if row[0]!='Precision ':
                    Pre.append(float(row[0]))
                    rec.append(float(row[1]))

                    F1.append(float(row[2]))
                    acc.append(float(row[3]))
                    loss.append(float(row[4]))
                    
                    if len(F1) == 10:
                        Pre_mean.append(statistics.mean(Pre))
                        rec_mean.append(statistics.mean(rec))

                        F1_mean.append(statistics.mean(F1))
                        acc_mean.append(statistics.mean(acc))
                        loss_mean.append(statistics.mean(loss))
                        Pre = []
                        rec = []

                        F1 = []
                        acc=[]
                        loss=[]

        out = open(output,'a', newline='')
        #print(F1_mean)
        csv_write = csv.writer(out,dialect='excel')
        csv_write.writerow(acc_mean)
        csv_write.writerow(Pre_mean)
        csv_write.writerow(rec_mean)
        csv_write.writerow(F1_mean)
        csv_write.writerow(loss_mean)

def mul_label_trans(csvfile,label,output):
    data = np.loadtxt(csvfile,delimiter=',')
    bl = np.loadtxt(label,delimiter=',')
    bl = bl.astype(int)
    print(bl[:,0])
    ml = data[bl[:,0]-1,19]
    print('ml shape = ',ml.shape)
    index = np.argwhere(bl[:,2]==0)
    np.savetxt(output,ml[index],delimiter=',')
    return 0

def get_mean_std(csvfile):
    data = np.loadtxt(csvfile,delimiter=',')
    n_fs = data.shape[1]
    mean = np.zeros(n_fs)
    std = np.zeros(n_fs)
    for i in range(n_fs):
        mean[i]=data[:,i].mean()
        std[i] =data[:,i].std()
    return mean, std
        
def get_class_report(report):
    report = report.split('\n')
    csv = np.zeros((11,3))
    for i in range(2,13):
        report[i]=report[i].split()
        for j in range(1,len(report[i])-1):
            csv[i-2][j-1]=float(report[i][j])
    print(csv)
    np.savetxt('2lfnn_classification_report.csv',csv,delimiter=',')

    

#process('FNN_results/dos_mitm_l2/test.csv','FNN_results/dos_mitm_l2/test_mean.csv')
#mul_label_trans('pcap file/mulabel_AN_3.csv','FNN_results/nondosl1(picked)/diff.csv','FNN_results/nondosl1(picked)/diff_mul.csv',)
m,s = get_mean_std(sys.argv[1])
print('mtm Ndos mean:',m)
print('mtm Ndos std:',s)

#np.savetxt('shared/mean.csv',m,delimiter=',')
#np.savetxt('shared/std.csv',s,delimiter=',')
#get_class_report(c)

#for macro and micro table average
def get_mai_table():
    file_list = ['lam.csv','fam.csv']
    for i in range(2):
        file = 'shared/'+file_list[i]
        m,s=get_mean_std(file)
        print(file_list[i],m,s)


# get mean for each model for bar plot
def get_mean_bar():
    path = 'shared/evaluation_results/'
    dic = ['ensemble/','lstm/','fnn/']
    files = ['precision.csv','recall.csv','f1.csv']
    for f in files:
        mean = np.zeros((3,11))
        for d in range(len(dic)):
            m,s = get_mean_std(path+dic[d]+f)
            mean[d,:]=m
        np.savetxt(path+f,mean,delimiter=',')


#get online testing packets number
def get_online_np():
    path = 'shared/label/'     
    for i in range(10):
        data = np.loadtxt(path+str(i)+'.csv',delimiter=',')
        label = data[:,-1]
        if i==0:
            labels = label.copy()
        else:
            labels = np.append(labels,label)
    print('total:',labels.shape[0])
    print('nondos',labels[(labels>0)&(labels<8)].shape[0]/labels.shape[0])
    print('dos',labels[labels>7].shape[0]/labels.shape[0])
#get_mai_table()
#get_online_np()
#os.environ['CUDA_VISIBLE_DEVICES']='1'
#model=load_model(sys.argv[1])
#print(model.summary())
