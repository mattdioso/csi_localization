#!/bin/sh
cd /jffs/CSI5G
mat_ext=".mat"
FILES=$(find /tmp -name 'capture*')
for f in $FILES
do
	file="${f#?????}"
	base="${file%?????}"
	mat_file=$base$mat_ext
	echo $mat_file
	/jffs/CSI5G/convertpcap.sh $file $mat_file
done
