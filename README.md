# random_mnist
DNN for MNIST dataset with random graph architecture

Usage:

- step1: python gengraph,oy <hidden layer 0> <hidden layer 1> ...

- step2: python gennn.py graph_<hidden layer 0><hidden layer 1> ... .dat

- step3: python3 randnn_<number of hidden nodes>_<number of edges>.py


Example:

% python3 gengraph.py 64 32

% python3 python3 gennn.py graph_64_32.dat

% python3 randnn_96_52544.py
