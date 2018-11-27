#!/bin/bash
num_file=$(ls -l Attack/ |grep "^-"|wc -l)
echo ${num_file}
for i in $( seq 1 ${num_file} )
do
tshark -r Attack/Attack${i}.pcap -Y mbtcp -T fields -e tcp.seq > Attack/CSV/Attack${i}.csv 
echo ${i}
done
#cat *.csv > full.csv          
#This is used to combine the csv files to one

