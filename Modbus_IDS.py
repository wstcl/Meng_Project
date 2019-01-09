import pyshark as pk
import numpy as np
from keras.models import load_model
import h5py
import time
feature_mean = []
feature_std = []
feature_mean = np.array(feature_mean)
feature_std = np.array(feature_std)
i = 0
model = load_model('lstm.h5')
capture = pk.LiveCapture('br0', display_filter = 'mbtcp')
capture.set_debug()
print('Modbus NIDS start, Press CTRL+C to stop')
while (True):
    for packet in capture.sniff_continuously():
        src_hash = 0
        dst_hash = 0
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
        if i == 0:
            raw = np.zeros((5,12))
        if i < 5:
            raw[i] = [src_hash,dst_hash,src_port,dst_port,seq_num,trans_ID,Func_Code,Ref_num,Reg_data,Excep,delta_time,stream_time]
            i = i + 1
        if i == 5:
            i = 0
            input = (raw -feature_mean)/feature_std
            input = input.reshape((1,5,12))
            label = model.predict(input)
            output = np.insert(raw,13,label,axis=1)
            if label.any() > 0.5:
                print(time.ctime(),"Dos attack detected")
            f=open('Log.csv','ab')
            np.savetxt(f,output,delimiter=",")
            f.close()






        

