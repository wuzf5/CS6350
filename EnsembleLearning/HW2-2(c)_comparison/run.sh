#!/bin/sh

# echo "running bagged tree algorithm for 1 to 500 bagging size, the result can be seen in the figure in the same folder"
python main.py --n_iters 500 --n_bags 100

# (SINGLE TREE) average bias w.r.t. all test samples:  0.10623151999999952
# (SINGLE TREE) average var w.r.t. all test samples:  0.06907523232323233
# (SINGLE TREE) general squared error w.r.t. test examples:  0.17530675232323184
# (BAGGED TREE) average bias w.r.t. all test samples:  0.12971048
# (BAGGED TREE) average var w.r.t. all test samples:  0.003734868686868687
# (BAGGED TREE) general squared error w.r.t. test examples:  0.13344534868686866