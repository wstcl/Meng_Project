import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support
def data_reshape(dst, label, timesteps):
    '''feature_mean = []
    feature_std = []
    for i in label:
            mean = dst[i].mean()
            std = dst[i].std()
            feature_mean.append(mean)
            feature_std.append(std)
            dst[i] = (dst[i] - mean) / std'''
    #reshape
    input = dst[label]
    Y = dst['result']
    input = np.array(input)
    Y = np.array(Y)
    Y = to_categorical(Y)
    drop = input.shape[0] % timesteps
    for i in range(drop):
        input = np.delete(input, input.shape[0] - i - 1, axis=0)
        Y = np.delete(Y, Y.shape[0] - i - 1, axis=0)
    X = input.reshape((input.shape[0] // timesteps, timesteps, input.shape[1]))       
    y = Y.reshape(Y.shape[0] // timesteps,timesteps,8)  #8 kinds of attack in dataset
    return X, y

def classreport(y_true, y_pred):
    # Count positive samples.
    c1 = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    c2 = np.sum(np.round(np.clip(y_pred, 0, 1)))
    c3 = np.sum(np.round(np.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision,recall,f1_score

def RNN(csvname):
    data = pd.read_csv(csvname)

    timestep = 100
    X_Label = [i for i in data.columns.tolist() if i not in 'result']
    X, y = data_reshape(data, X_Label, timestep)
    precision = []
    recall = []
    f1 = []
    for kfold in range(2):
        model = Sequential()
        model.add(LSTM(256,return_sequences=True,input_shape=(X.shape[1],X.shape[2])))
        model.add(LSTM(128,return_sequences=True))
        model.add(Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        print(model.summary())
        model.fit(X, y, epochs=30, batch_size=300)
        preds_prob = preds.reshape(preds.shape[0]*timestep,8)
        pred_class = np.argmax(preds_prob, axis=1)
        #model.save('lstm.h5')
        pred_index = np.where(pred_class == 6)
        real_index = np.where(data['result'] == 6)
        print(precision_recall_fscore_support(data['result'],pred_class,average='macro'))
    
RNN("water_final.csv")
