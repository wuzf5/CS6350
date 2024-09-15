#!/bin/sh
max_max_depth=16
purity_measures=('entropy' 'majority' 'gini')
test_set=('train' 'test')

echo "-----------------------------For Question 3(a)-----------------------------"
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
echo "-----------------------------For Question 3(b)-----------------------------"
for dataset in "${test_set[@]}";
do
    for measure in "${purity_measures[@]}";
    do
        for max_depth in `seq ${max_max_depth}`;
        do
            echo "max depth is ${max_depth}, purity measure is ${measure}, tested on ${dataset}set:"
            python3 complete_unknown.py --max_depth ${max_depth} --purity_measure ${measure} --test_set ${dataset}
        done
    done
done