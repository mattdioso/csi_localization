#!/bin/sh

mat_ext=".mat"

cd /jffs/CSI5G

for file in /tmp/*.pcap;
do
	if [ -f "$file" ]; then
		pcap_file="${file#?????}"
		pcap="${file%?????}"
		filename="${pcap#?????}"
		mat_file="$filename$mat_ext"
		echo $pcap_file $mat_file
		/jffs/CSI5G/convertpcap.sh $pcap_file $mat_file
	fi
done
