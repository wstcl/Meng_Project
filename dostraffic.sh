#!/bin/bash
while [ true ]
do
	IP="10.0.0.5"
	DIV=$((7))
	DIV1=$((4))
	R1=$(($RANDOM%$DIV1))
	R=$(($RANDOM%$DIV))
	if [ $R -eq 0 ]; then
		if [ $R1 -eq 0 ]; then
		#Incorrect CRC
			if modpoll -r 52210 -0 -1 $IP > \dev\null 2>&1; then
			 for i in {1..300};
			 do
			  modpoll -m enc -t 4 -0 -1 -r 32210 -l 1 $IP > \dev\null 2>&1;
			 done	  
			 until modpoll -r 52211 -0 -1 $IP > \dev\null 2>&1
				do
				echo no
				done
			fi
		elif [ $R1 -eq 1 ]; then
		#scan
			if modpoll -r 52210 -0 -1 $IP > \dev\null 2>&1; then
			for i in {1..1000};
			do
			 modpoll -t 4 -r 42210 -0 -1 -l 1 $IP;
			done
			 until modpoll -r 52211 -0 -1 $IP > \dev\null 2>&1
			  do
			   echo no
			  done
			fi
		fi
	elif [ $R -gt 0 ]; then
		if [ $R1 -eq 0 ]; then
		#normal CRC
			timeout $(($RANDOM%2+1)) modpoll -m enc -t 4 -0 -r 32210 $IP;
		elif [ $R1 -eq 1 ]; then
		#normal scan
			timeout $(($RANDOM%2+1)) modpoll -r 42210 -t 4 -0 $IP;
		fi
	fi
done
