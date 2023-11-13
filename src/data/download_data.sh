#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mOEP6woH_9Ox8_Va0V5DibWttYR6u930' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mOEP6woH_9Ox8_Va0V5DibWttYR6u930" -O data/raw/train_dataset.zip && rm -rf /tmp/cookies.txt
