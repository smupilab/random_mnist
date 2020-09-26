import matplotlib.pyplot as pyplot
import networkx as nx
import random
import sys

num_inputs = 784
num_outputs = 10

hidden_layers = sys.argv[1:]

input_name = "X"
out_fname = "graph_{}.txt".format("_".join(hidden_layers))
output = open(out_fname, "w")


G = nx.DiGraph()

# input layer
for u in range(num_inputs):
    for v in range(int(hidden_layers[0])):
        G.add_edge("X_{}".format(u), "h_0_{}".format(v))

# hidden layers
for l in range(len(hidden_layers)-1):
    for u in range(int(hidden_layers[l])):
        for v in range(int(hidden_layers[l+1])):
            G.add_edge("h_{}_{}".format(l, u), "h_{}_{}".format(l+1, v))

# output layer
for u in range(int(hidden_layers[-1])):
    for v in range(num_outputs):
        G.add_edge("h_{}_{}".format(len(hidden_layers)-1, u), "O_{}".format(v))
#-------------------------------

output.write("### num_inputs, num_outputs, num_hiddennodes, num_edges\n")
output.write("### {} {} {} {}\n".format(num_inputs, num_outputs, G.number_of_nodes() - num_inputs - num_outputs, G.number_of_edges()))

for edge in G.edges:
    output.write("{} {}\n".format(edge[0], edge[1]))

print("Graph generated: {}".format(out_fname))

out_pickle = "graph_{}.dat".format("_".join(hidden_layers))
nx.write_gpickle(G, out_pickle)
