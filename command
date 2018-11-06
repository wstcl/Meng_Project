tshark -i br0 -a duration:1800 -Y mbtcp -T fields -E aggregator=' ' -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e tcp.seq -e mbtcp.trans_id -e modbus.func_code -e modbus.reference_num -e modbus.register.uint16 -e modbus.exception_code -e frame.time_epoch -e tcp.relative_time > dos.csv


#open this file here and save it as text format(change sth and undo)
#When in Windows, use notepad to open it then save it as utf8 format.
