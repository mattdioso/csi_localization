#!/bin/sh

ip_addr=$(ifconfig br0 | sed -rn 's/^.*inet addr:(([0-9]+\.){3}[0-9]+).*$/\1/p')
echo $ip_addr

curl -X POST https://textbelt.com/text --data-urlencode phone='9255221652' --data-urlencode message='ip_addr' -d key=textbelt
