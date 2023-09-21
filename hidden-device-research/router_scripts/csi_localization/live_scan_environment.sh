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
		/jffs/CSI5G/tcpdump -i eth7 port 5500 -vv -w /tmp/small_${i}_$((2**x))_undense_trace.pcap -c 4
		x=$(( x+1 ))
		echo "switching antennas"
	done

	i=$(( i +1 ))
	read -p "Move it to the next grid and press enter" < /dev/tty
done
echo "all done"
