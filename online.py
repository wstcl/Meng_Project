import pyshark as pk
import numpy as np
capture = pk.LiveCapture(interface='br0',display_filter='mbtcp')
print('Start capture br0')
for packet in capture.sniff_continuously(packet_count=10):
    src_ip = packet['ip'].src
    dst_ip = packet['ip'].dst
    src_port = packet.tcp.srcport
    dst_port = packet.tcp.dstport
    stream_index = packet.tcp.stream
    seq_num = packet.tcp.seq
    trans_ID = packet.mbtcp.trans_id
    Func_Code = packet.modbus.func_code
   # Ref_num = packet.modbus.reference_num
    #if packet['mbtcp'].modbus.regval_uint16:
    #    Reg_data = packet.modbus.regval_uint16
    #Excep = packet.modbus.exception_code
    #delta_time = packet.time_delta_displayed
    stream_time = packet.tcp.time_relative
    print('Just arrived', packet)
print('Cap done')

