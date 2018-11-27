tshark -r traffic/ddd_00001_20181125185341.pcap -Y mbtcp -T fields -E aggregator=' ' -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e tcp.stream -e tcp.seq -e mbtcp.trans_id -e modbus.func_code -e modbus.reference_num -e modbus.regval_uint16 -e modbus.exception_code -e tcp.time_relative -e frame.time_delta_displayed > NOV25_7.csv

#open this file here and save it as text format(change sth and undo)
#When in Windows, use notepad to open it then save it as utf8 format.

