#!/bin/sh
purity_measures=('entropy' 'majority' 'gini')
max_max_depth=6
test_set=('train' 'test') # which dataset to test on

for dataset in "${test_set[@]}";
do
    for measure in "${purity_measures[@]}";
    do
        for max_depth in `seq ${max_max_depth}`;
        do
            echo "max depth is ${max_depth}, purity measure is ${measure}, tested on ${dataset}set:"
            python3 main.py --max_depth ${max_depth} --purity_measure ${measure} --test_set ${dataset}
        done
    done
done