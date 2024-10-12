#!/bin/sh

# echo "running bagged tree algorithm for 1 to 500 bagging size, the result can be seen in the figure in the same folder"
python main.py --n_forests 100 --n_trees_per_forest 500 --G 4

# (SINGLE TREE) average bias w.r.t. all test samples:  0.11926732000000027
# (SINGLE TREE) average var w.r.t. all test samples:  0.05843301010101011
# (SINGLE TREE) general squared error w.r.t. test examples:  0.17770033010101038
# (RANDOM FOREST) average bias w.r.t. all test samples:  0.13936767999999997
# (RANDOM FOREST) average var w.r.t. all test samples:  0.002456888888888889
# (RANDOM FOREST) general squared error w.r.t. test examples:  0.14182456888888886