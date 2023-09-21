#!/bin/sh

cd /jffs/CSI5G

START=$(date +%s)

echo $1 $2

x=0
while [ ${x} -le 3 ];
do
	echo "/jffs/CSI5G/config5GHz.sh 36 160 $((2**x)) 1 8"
	echo "/tmp/capture_$1in_$2_$((2**x))_trace.pcap"
	/jffs/CSI5G/config5GHZ.sh 36 160 $((2**x)) 1 8
	/jffs/CSI5G/setmacfilter.sh eth7 88:00 00:1f:54:91:1b:3a
	/jffs/CSI5G/tcpdump -i eth7 port 5500 -vv -w /tmp/capture_$1in_$2_$((2**x))_trace.pcap -c 4 
	/jffs/CSI5G/convertpcap.sh capture_$1in_$2_$((2**x))_trace.pcap capture_$1in_$2_$((2**x))_trace.mat
	x=$(( x+1 ))
	echo "switching antennas"
done

END=$(date +%s)
DIFF=$(($END-$START))
echo "took $DIFF seconds"
