import pyshark as pk
import numpy as np
capture = pk.LiveCapture('br0', display_filter = 'mbtcp')
capture.set_debug()
print('Modbus NIDS start, Press CTRL+C to stop')
while (True):
    for packet in capture.sniff_continuously():
        src_ip = packet.ip.src
        dst_ip = packet.ip.dst
        src_port = packet.tcp.srcport
        dst_port = packet.tcp.dstport
        stream_index = packet.tcp.stream
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
        

