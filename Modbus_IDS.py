import pyshark as pk
import numpy as np
from keras.models import load_model
import h5py
import time
feature_mean = [1485969.6506835904, 2696469.353480891, 18865.4025525368, 28989.058054477428, 57268.50361900766, 8831.50869261163, 4.708689115004021, 18519.598052379453, 25.74041050386377, 0.20607014231266826, 0.0092014301653904, 482.78965815695653]
feature_std = [2285964.50843903, 2765235.7987897503, 23168.549619500987, 23445.339534367795, 95452.5125702013, 12625.33563010513, 5.402296630386151, 19521.437757365704, 43.72046432971252, 0.40448215143340616, 0.01920430014961369, 795.0218303168346]
feature_mean = np.array(feature_mean)
feature_std = np.array(feature_std)
timestep = 10
i = 0
model = load_model('Lstm.h5')
csvname = 'MitM.csv'
flag = 0
capture = pk.LiveCapture('br0', display_filter = 'mbtcp')
label_index = 14
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
                FLP = np.array([src_hash,dst_hash,src_port,dst_port,seq_num,trans_ID,Func_Code,Ref_num,Reg_data,Excep,delta_time,stream_time,eth_src,eth_dst,0])
                np.savetxt(f,FLP,delimiter=",")
                f.close()
                continue
            #check if this packet is a flag response
            if flag == 1:
                if packet.tcp.stream == flag_index:
                    continue
            #if not a flag, fit into lstm model
            raw[i] = [src_hash,dst_hash,src_port,dst_port,seq_num,trans_ID,Func_Code,Ref_num,Reg_data,Excep,delta_time,stream_time,eth_src,eth_dst]
            i = i + 1
        if i == timestep:
            i = 0
            input = (raw[:,0:12] -feature_mean)/feature_std
            input = input.reshape((1,timestep,12))
            label = model.predict(input) 
            label[label>0.5]=1
            label[label<=0.5]=0
            output = np.insert(raw,label_index,label,axis=1)
            if label.any() == 1:
                print(time.ctime(),"Dos attack detected")
                print(label)
            f=open(csvname,'ab')
            np.savetxt(f,output,delimiter=",")
            f.close()

