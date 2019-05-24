from keras.models import load_model
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
path = 'shared/models/'
mean = np.loadtxt(path+'mean.csv',delimiter=',')
std = np.loadtxt(path + 'std.csv',delimiter=',')
fnn = load_model(path+'FNN.h5')
lstm = load_model(path+'LSTM.h5')
timestep = 10
n_classes=11


def evaluate_a(pre,true):
    pre= pre
    true=true
    p = precision_score(y_true=true,y_pred=pre,average='macro')
    r = recall_score(y_true=true,y_pred=pre,average='macro')
    f = f1_score(y_true=true,y_pred=pre,average='macro')
    a = accuracy_score(y_true=true,y_pred=pre)
    return p,r,f,a

def evaluate_i(pre,true):
    pre=pre
    true=true
    p = precision_score(y_true=true,y_pred=pre,average='micro')
    r = recall_score(y_true=true,y_pred=pre,average='micro')
    f = f1_score(y_true=true,y_pred=pre,average='micro')
    a = accuracy_score(y_true=true,y_pred=pre)
    return p,r,f,a
 
def evaluate_n(pre,true):
    pre=pre
    true=true
    p = precision_score(y_true=true,y_pred=pre,average=None)
    r = recall_score(y_true=true,y_pred=pre,average=None)
    f = f1_score(y_true=true,y_pred=pre,average=None)
    return p,r,f

precision_l = np.zeros((10,11))
recall_l = np.zeros((10,11))
f1_l = np.zeros((10,11))
precision_f = np.zeros((10,11))
recall_f = np.zeros((10,11))
f1_f = np.zeros((10,11))



for i in range(10):
    data = np.loadtxt('shared/label/'+str(i)+'.csv',delimiter=',')
    drop = data.shape[0]%timestep
    for j in range(drop):
        data = np.delete(data,data.shape[0]-j-1,axis=0)
    X = data[:,:-1]
    X[:,11]=X[:,11]%3000
    X = (X-mean)/std
    input_lstm = X.reshape((X.shape[0]//timestep,timestep,X.shape[1]))
    label_lstm = lstm.predict(input_lstm)
    label_lstm = label_lstm.reshape((label_lstm.shape[0]*timestep,n_classes))
    label_lstm = np.argmax(label_lstm,axis=1)

    label_fnn = fnn.predict_classes(X)
    y = data[:,-1]
    print(y.shape)
    pil,ril,fil,ail = evaluate_i(label_lstm,y)
    pal,ral,fal,aal = evaluate_a(label_lstm,y)
    pif,rif,fif,aif = evaluate_i(label_fnn,y)
    paf,raf,faf,aaf = evaluate_a(label_fnn,y)
    print(ail,',',pil,',',ril,',',fil,file=open('shared/li.csv','a'))
    print(aal,',',pal,',',ral,',',fal,file=open('shared/la.csv','a'))
    print(aif,',',pif,',',rif,',',fif,file=open('shared/fi.csv','a'))
    print(aaf,',',paf,',',raf,',',faf,file=open('shared/fa.csv','a'))
    '''
    pl,rl,fl = evaluate_n(label_lstm,y)
    pf,rf,ff = evaluate_n(label_fnn,y)
    precision_l[i,:]=pl 
    recall_l[i,:]=rl
    f1_l[i,:]=fl
    precision_f[i,:] = pf
    print(precision_f)
    recall_f[i,:]=rf
    f1_f[i,:]=ff
np.savetxt('shared/evaluation_results/lstm/precision.csv',precision_l,delimiter=',')
np.savetxt('shared/evaluation_results/lstm/recall.csv',recall_l,delimiter=',')
np.savetxt('shared/evaluation_results/lstm/f1.csv',f1_l,delimiter=',')
np.savetxt('shared/evaluation_results/fnn/precision.csv',precision_f,delimiter=',')
np.savetxt('shared/evaluation_results/fnn/recall.csv',recall_f,delimiter=',')
np.savetxt('shared/evaluation_results/fnn/f1.csv',f1_f,delimiter=',')
'''




