import random
import sys

def cycle(u):
    global G
    global visited

    if visited[u]:
        return True
    else:
        visited[u] = True

    for v in range(num_nodes):
        if G[u][v] == True: # edge (u, v)
            if cycle(v):
                return True
    return False

if len(sys.argv) != 3:
    print("Usage: python gengraph.py <number of hidden nodes> <number of edges>")
    quit()


num_inputs = 784
num_outputs = 10
num_hiddennodes = int(sys.argv[1])
num_ionodes = num_inputs + num_outputs
num_nodes = num_inputs + num_outputs + num_hiddennodes
num_edges = int(sys.argv[2])

input_name = "X"

out_fname = "graph_{}_{}.txt".format(num_hiddennodes, num_edges)
output = open(out_fname, "w")

output.write("## num_inputs, num_outputs, num_hiddennodes, num_edges\n")
output.write("{} {} {} {}\n".format(num_inputs, num_outputs, num_hiddennodes, num_edges))

edges = 0
G = [[False]*num_nodes for _ in range(num_nodes)]

for i in range(num_outputs):
    u = random.randrange(num_inputs)
    v = i + num_inputs
    G[u][v] = True
    output.write("{} {}\n".format(u, v))
    edges = edges + 1
 
while(edges < num_edges):
    u = random.randrange(num_nodes)
    v = random.randrange(num_nodes)
    if (v < num_inputs):
        continue
    if (u >= num_inputs and u < num_ionodes):
        continue
    G[u][v] = True
    visited = [False]*num_nodes
    if not cycle(u):
        output.write("{} {}\n".format(u, v))
        edges = edges + 1
##        print("{} edges have been generated".format(edges))
    else:
        G[u][v] = False
##        print("({} {}) edge tried".format(u, v))
  

output.close()
print("Graph generated: {}".format(out_fname))
