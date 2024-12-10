#!/bin/sh
widths=("5" "10" "25" "50" "100")

for width in "${widths[@]}";
do
    echo "width is ${width}:"
    python3 main.py --width ${width}
done