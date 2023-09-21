#!/bin/sh

#the extract function appears to only work for smaller sets of captures at a time (400)
#produced file format is env identifier_L_W_H_grid_antenna_hidden or visible_num_trace.pcap

cd /jffs/CSI5G

i=1
while [ ${i} -le 21 ];
do
	x=0
	while [ ${x} -le 3 ]; 
	do
		#echo $((2**x))
		echo "/jffs/CSI5G/config5GHz.sh 36 160 $((2**x)) 1 8"
		/jffs/CSI5G/config5GHZ.sh 36 160 $((2**x)) 1 8
		/jffs/CSI5G/setmacfilter.sh eth7 88:00 00:1f:54:91:1b:3a
		iter=1
		while [ ${iter} -le 5 ];
		do
			/jffs/CSI5G/tcpdump -i eth7 port 5500 -vv -w /tmp/320_10.6_10.6_3.3_${i}_$((2**x))_los_${iter}_trace.pcap -c 400
			iter=$(( iter+1 ))
		done
		x=$(( x+1 ))
		echo "switching antennas"
	done
	read -p "hide the device now and press enter" < /dev/tty
	x=0
	while [ ${x} -le 3 ];
	do
		echo "/jffs/CSI5G/config5GHz.sh 36 160 $((2**x)) 1 8"
		/jffs/CSI5G/config5GHZ.sh 36 160 $((2**x)) 1 8
		/jffs/CSI5G/setmacfilter.sh eth7 88:00 00:1f:54:91:1b:3a
		#read -p "hide the device now and press enter" < /dev/tty
		iter=1
		while [ ${iter} -le 5 ];
		do
			/jffs/CSI5G/tcpdump -i eth7 port 5500 -vv -w /tmp/320_10.6_10.6_3.3_${i}_$((2**x))_nlos_${iter}_trace.pcap -c 400
			iter=$(( iter+1 ))
		done
		x=$(( x+1 ))
		echo "switching antennas"
		#read -p "uncover the camera and press enter" < /dev/tty
	done

	i=$(( i +1 ))
	read -p "Move it to the next grid and press enter" < /dev/tty
done
echo "all done"
