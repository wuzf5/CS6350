#!/bin/sh

# echo "running bagged tree algorithm for 1 to 500 bagging size, the result can be seen in the figure in the same folder"
python3 main.py --n_iters 500 --n_bags 100


# (SINGLE TREE) average bias w.r.t. all test samples:  0.09722823999999958
# (SINGLE TREE) average var w.r.t. all test samples:  0.032246222222222225
# (SINGLE TREE) general squared error w.r.t. test examples:  0.1294744622222218
# (BAGGED TREE) average bias w.r.t. all test samples:  0.11763572000000004
# (BAGGED TREE) average var w.r.t. all test samples:  0.0010023030303030307
# (BAGGED TREE) general squared error w.r.t. test examples:  0.11863802303030307