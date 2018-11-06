#!/bin/bash
while [ true ]
do
	IP="10.0.0.5"
	DIV=$((7))
	DIV1=$((4))
	R1=$(($RANDOM%$DIV1))
	R=$(($RANDOM%$DIV))
	if [ $R -gt 0 ]; then
		if [ $R1 -eq 0 ]; then
		#Incorrect CRC
			for i in {1..100};
			do
			 modpoll -m enc -t 3 -0 -r 32210 -l 1 $IP;
			done
		elif [ $R1 -eq 1 ]; then
		#scan
			for i in {1..100};
			do
			 modpoll -r 42210 -c 5 -1 -l 1 $IP;
			done
		fi
	elif [ $R -eq 0 ]; then
		if [ $R1 -eq 0 ]; then
		#normal CRC
			timeout 4 modpoll -m enc -t 3 -0 -r 32210 $IP;
		elif [ $R1 -eq 1 ]; then
		#normal scan
			modpoll -r 42210 -c 10 -1 $IP;
		fi
	fi
done