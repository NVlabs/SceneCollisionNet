#!/bin/bash

export fileid=1lz8N2kMsfgrZf2PVmbpq5mxBzqmQHUdL
export filename=weights.zip

wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

unzip weights.zip
rm weights.zip cookies.txt confirm.txt