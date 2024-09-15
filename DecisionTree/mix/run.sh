#!/bin/sh
purity_measures=('entropy' 'majority' 'gini')
max_max_depth=16

echo "For Question 3(a)"
for max_depth in `seq ${max_max_depth}`;
do
    for measure in "${purity_measures[@]}";
    do
        echo "max depth is ${max_depth}, purity measure is ${measure}:"
        python3 main.py --max_depth ${max_depth} --purity_measure ${measure}
    done
done

echo "For Question 3(b)"
for max_depth in `seq ${max_max_depth}`;
do
    for measure in "${purity_measures[@]}";
    do
        echo "max depth is ${max_depth}, purity measure is ${measure}:"
        python3 complete_unknown.py --max_depth ${max_depth} --purity_measure ${measure}
    done
done