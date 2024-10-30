#!/bin/sh

# echo "running bagged tree algorithm for 1 to 500 bagging size, the result can be seen in the figure in the same folder"
python main.py --n_forests 100 --n_trees_per_forest 500 --G 6

# (SINGLE TREE) average bias w.r.t. all test samples:  0.11072154000000052
# (SINGLE TREE) average var w.r.t. all test samples:  0.06440450505050506
# (SINGLE TREE) general squared error w.r.t. test examples:  0.17512604505050558
# (RANDOM FOREST) average bias w.r.t. all test samples:  0.13360770000000002
# (RANDOM FOREST) average var w.r.t. all test samples:  0.0016952525252525255
# (RANDOM FOREST) general squared error w.r.t. test examples:  0.13530295252525254