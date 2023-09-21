#!/bin/sh

cd /jffs/CSI5G

i=0
while [ ${i} -ne 4 ];
do
	echo "relifing"
	/jffs/CSI5G/relife.sh eth7
	sleep 3
done
