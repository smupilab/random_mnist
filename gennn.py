import random
import sys
import queue

if len(sys.argv) != 2:
    print("Usage: python genn.py <graph file name>")
    quit()

batch_size = 100
numepochs = 20

graph_fname = sys.argv[1]
graph = open(graph_fname, "r");

lines = graph.readlines()
num_inputs, num_outputs, num_hiddennodes, num_edges = lines[1].split(" ")
num_inputs = int(num_inputs)
num_outputs = int(num_outputs)
num_hiddennodes = int(num_hiddennodes)
num_edges = int(num_edges)
num_nodes = num_inputs + num_outputs + num_hiddennodes
g = [[-1]*num_nodes for _ in range(num_nodes)]

for i in range(num_edges):
    u, v = lines[i+2].split(" ")
    g[int(u)][int(v)] = i

inedges = [[] for _ in range(num_nodes)]
for u in range(num_nodes):
    for v in range(num_nodes):
        if g[u][v] != -1:
            inedges[v].append(u)

out_fname = "randnn_{}_{}.py".format(num_hiddennodes, num_edges)
log_fname = "randnn_{}_{}.log".format(num_hiddennodes, num_edges)
output = open(out_fname, "w")

output.write("import tensorflow as tf\n")
output.write("\n")
output.write("num_classes = 10\n")
output.write("img_rows, img_cols = 28, 28\n")
output.write("num_channels = 1\n")
output.write("input_shape = (img_rows, img_cols, num_channels)\n")
output.write("\n")
output.write("(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()\n")
output.write("x_train, x_test = x_train / 255.0, x_test / 255.0\n")
output.write("\n")
output.write("log = open('{}','w')\n".format(log_fname))
output.write("\n")
output.write("print('Training started')\n")
output.write("\n")
output.write("### num_inputs, num_outputs, num_hiddennodes, num_edges\n")
output.write("### {} {} {} {}\n".format(num_inputs, num_outputs, num_hiddennodes, num_edges))
output.write("\n")
output.write("model_input = tf.keras.layers.Input(shape=input_shape)\n")
output.write("flat_input = tf.keras.layers.Flatten()(model_input)\n")
output.write("\n")

# input
for i in range(num_inputs):
    output.write("out{} = tf.gather(flat_input, [{}], axis=1)\n".format(i, i))
output.write("\n")

# hidden nodes (via BFS)
q = queue.Queue()
buf = []
for i in range(num_outputs):
    q.put(i + num_inputs) # output nodes
while(not q.empty()):
    v = q.get() 
    if len(inedges[v]) > 0:
        line = "out{} = tf.keras.layers.Dense(1, activation='relu')(tf.concat([".format(v)
        # edge u -> v
        for u in range(len(inedges[v])-1):
            line += "out{}, ".format(inedges[v][u])
            q.put(inedges[v][u])
            ## print("q.put {}".format(inedges[v][u]))
        line += "out{}], 1))\n".format(inedges[v][-1])
        q.put(inedges[v][-1])
        buf.append(line)

while(len(buf) > 0):
    output.write(buf.pop())
output.write("\n")

# final layer
output.write("final_nodes = tf.concat([")
for i in range(num_outputs - 1):
    v = i + num_inputs    # output node number
    output.write("out{}, ".format(v))
output.write("out{}], 1)\n".format(num_inputs + num_outputs - 1))
output.write("\n")

output.write("final_output = tf.nn.softmax(final_nodes, axis=1)\n")
output.write("\n")
output.write("model = tf.keras.Model(model_input, final_output)\n")
output.write("model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n")
output.write("model.summary()\n")
output.write("print('Training started')\n")
output.write("model.fit(x_train, y_train, epochs={}, verbose=1, validation_data=(x_test, y_test))\n".format(numepochs))
output.write("test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)\n")
output.write("print(test_acc)\n")
output.write("log.write('Accuracy = {}\\n'.format(test_acc))\n")
output.write("log.close()\n")
output.close()
