import pyshark as pk
import numpy as np
from keras.models import load_model
import h5py
import os
import time
from keras.utils import to_categorical

feature_mean = np.loadtxt('mean.csv',delimiter=',')
feature_std = np.loadtxt('std.csv',delimiter=',')
timestep = 10
i = 0
j = 0
turn = 0
n_classes = 11
fnn = load_model('models/FNN.h5')
lstm = load_model('models/LSTM.h5')
ensemble = load_model('models/ensemble.h5')
tm = time.ctime()+'/'
os.mkdir(tm)
csvname = tm+'all_classes.csv'
flag = 0
capture = pk.LiveCapture('br0', display_filter = 'mbtcp')
label_index = 21
print('Modbus NIDS start, Press CTRL+C to stop')
while (True):
    if j ==15:
        break
    for packet in capture.sniff_continuously():
        src_hash = 0
        dst_hash = 0
        HH = 0
        LL = 0
        H = 0
        L = 0
        speed = 0
        t1 = 0
        t2 = 0
        src_ip = packet.ip.src
        src_ip = src_ip.split('.')
        for hash in src_ip:
            src_hash = 31*src_hash + int(hash)
        dst_ip = packet.ip.dst
        dst_ip = dst_ip.split('.')
        for hash in dst_ip:
            dst_hash = 31*dst_hash + int(hash)
        src_port = packet.tcp.srcport
        dst_port = packet.tcp.dstport
        seq_num = packet.tcp.seq
        trans_ID = packet.mbtcp.trans_id
        try:
            Func_Code = packet.modbus.func_code
        except AttributeError:
            Func_Code = 0
        try:
            Ref_num = packet.modbus.reference_num
        except AttributeError:
            Ref_num = 0
        try:
            Reg_data = packet.modbus.regval_uint16
        except AttributeError:
            Reg_data = 0
        try:
            Excep = packet.modbus.exception_code
        except AttributeError:
            Excep = 0
        delta_time = packet.frame_info.time_delta_displayed
        stream_time = packet.tcp.time_relative
        if src_port == '502':
            Ref_num = 0
        if Ref_num == '32210':
            speed = Reg_data
        if Ref_num == '42210':
            t1 = Reg_data
        if Ref_num == '42211':
            t2 = Reg_data
        if Ref_num == '42212':
            HH = Reg_data
        if Ref_num == '42213':
            LL = Reg_data
        if Ref_num == '42214':
            H = Reg_data
        if Ref_num == '42215':
            L = Reg_data

        eth_src = packet.eth.src
        eth_src = eth_src.replace(':','')
        eth_src = int(eth_src,16)
        eth_dst = packet.eth.dst
        eth_dst = eth_dst.replace(':','')
        eth_dst = int(eth_dst,16)
        
        if i == 0:
            raw = np.zeros((timestep,label_index))
        if i < timestep:
            #check if this packet is a flag, if yes, record and go to the next loop
            if Ref_num == 52210 or Ref_num ==52211:
                flag = 1
                flag_index=packet.tcp.stream
                f=open(csvname,'ab')
                FLP = np.array([src_hash,dst_hash,src_port,dst_port,seq_num,trans_ID,Func_Code,Ref_num,Reg_data,Excep,delta_time,stream_time,HH,LL,H,L,speed,t1,t2,eth_src,eth_dst,0])
                np.savetxt(f,FLP,delimiter=",")
                f.close()
                continue
            #check if this packet is a flag response
            if flag == 1:
                if packet.tcp.stream == flag_index:
                    continue
            #if not a flag, fit into lstm model
            raw[i] = [src_hash,dst_hash,src_port,dst_port,seq_num,trans_ID,Func_Code,Ref_num,Reg_data,Excep,delta_time,stream_time,HH,LL,H,L,speed,t1,t2,eth_src,eth_dst] #need to be raw[i]
            i = i + 1
        if i == timestep:
            i = 0
            input = raw[:,0:19] 
            input[:,11]=input[:,11]%3000 
            input = (input-feature_mean)/feature_std
            input_lstm = input.reshape((1,timestep,19))
            label_lstm = lstm.predict(input_lstm)
            label_lstm = label_lstm.reshape((timestep,n_classes))
            label_lstm = np.argmax(label_lstm,axis=1)
            label_lstm = to_categorical(label_lstm,num_classes=n_classes)

            label_fnn = fnn.predict_classes(input)
            label_fnn = to_categorical(label_fnn,num_classes=n_classes) 
            ensemble_input = np.dstack((label_lstm,label_fnn))
            ensemble_input = np.reshape(ensemble_input,(ensemble_input.shape[0],ensemble_input.shape[1]*ensemble_input.shape[2]))
            label=ensemble.predict_classes(ensemble_input)
            output = np.insert(raw,label_index,label,axis=1)
            if label.any() > 0:
                print(time.ctime(),"Attack detected")
                print(label)
            f=open(csvname,'ab')
            np.savetxt(f,output,delimiter=",")
            f.close()
            if os.path.getsize(csvname)>60000000 and j<6:
                j = j+1
                csvname = tm+str(j)+'.csv'
