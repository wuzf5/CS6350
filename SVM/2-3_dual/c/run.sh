#!/bin/sh
Cs=("100/873" "500/873" "700/873")
gammas=("0.1" "0.5" "1" "5" "100")

for C in "${Cs[@]}";
do
    for gamma in "${gammas[@]}";
    do
        echo "C is ${C}, gamma is ${gamma}"
        Cvalue=$(echo "scale=4; $C" | bc)
        gammavalue=$(echo "scale=4; $gamma" | bc)
        python3 main.py --C ${Cvalue} --gamma ${gammavalue}
    done
done

echo "Start computing overlapping numbers of support vectors"
python3 compute_overlap.py