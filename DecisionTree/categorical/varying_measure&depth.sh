#!/bin/sh
purity_measures=('entropy' 'majority' 'gini')
max_max_depth=6

for max_depth in `seq ${max_max_depth}`;
do
    for measure in "${purity_measures[@]}";
    do
        echo "max depth is ${max_depth}, purity measure is ${measure}:"
        python ID3.py --max_depth ${max_depth} --purity_measure ${measure}
    done
done