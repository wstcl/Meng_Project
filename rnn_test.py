import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical

def data_reshape(dst, label, timesteps):
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
    y = Y.reshape(Y.shape[0] // timesteps, timesteps*8)
    return X, y

def RNN(csvname):
    data = pd.read_csv(csvname)

    timestep = 1
    X_Label = [i for i in data.columns.tolist() if i not in 'result']
    X, y = data_reshape(data, X_Label, timestep)

    for kfold in range(1):
        model = Sequential()
        model.add(LSTM(256,return_sequences=True,input_shape=(X.shape[1],X.shape[2])))
        model.add(LSTM(128))
        model.add(Dense(8*timestep, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        print(model.summary())
        model.fit(X, y, epochs=20, batch_size=1000)
        preds = model.predict(X)

        preds_prob = preds.reshape(preds.shape[0]*timestep,8)
        pred_class = np.argmax(preds_prob, axis=1)
        #model.save('lstm.h5')
        pred_index = np.where(pred_class == 6)
        real_index = np.where(data['result'] == 6)
        fp = len(set(real_index[0])- set(pred_index[0]))/len(pred_index[0])
        print(fp)
    print("Done")
RNN("D:\\textbooks\\SCADA_PAPER\\water_final.csv")