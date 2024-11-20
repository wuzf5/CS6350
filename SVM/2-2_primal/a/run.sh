#!/bin/sh
Cs=("100/873" "500/873" "700/873")

for C in "${Cs[@]}";
do
    echo "C is ${C}:"
    value=$(echo "scale=4; $C" | bc)
    python3 main.py --C ${value}
done