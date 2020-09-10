import random
import sys

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

for i in range(num_inputs + num_outputs, num_nodes):
    output.write("def node{}(X):\n".format(i))
    v = i
    output.write("    result = B[{}]".format(v))
    for u in range(num_nodes):
        if g[u][v] != -1: # if there is an edge (u, v)
            if u < num_inputs: # the hidden node at the first layer
                output.write(" + tf.gather(X, {}, axis=1)*W[{}]".format(u, g[u][v]))
            else:
                output.write(" + node{}(X)*W[{}]".format(u, g[u][v]))
    output.write("\n")
    output.write("    return(tf.nn.relu(result))\n\n")


output.write("@tf.function\n")
output.write("def Hypothesis(X):\n")
for i in range(num_outputs):
    v = i + num_inputs    # output node number
    output.write("    out{} = B[{}]".format(i, v))
    for u in range(num_nodes):
        if g[u][v] != -1: # if there is an edge (u, v)
            if u < num_inputs: # the hidden node at the first layer
                output.write(" + tf.gather(X, {}, axis=1)*W[{}]".format(u, g[u][v]))
                ##output.write(" + tf.matmul(tf.gather(X, {}, axis=0), tf.gather(W, {}, axis=0))".format(u, g[u][v]))
            else:
                output.write(" + node{}(X)*W[{}]".format(u, g[u][v]))
    output.write("\n")
output.write("    result = tf.stack([")
for i in range(num_outputs - 1):
    output.write("out{}, ".format(i))
output.write("out{}], axis=1)\n".format(num_outputs - 1))
output.write("    return(result)\n")

output.write("final_output = tf.keras.layers.Dense(num_classes, activation='softmax')({})\n".format(zzzz))
output.write("\n")
output.write("model = tf.keras.Model(model_input, final_output)\n")
output.write("model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n")
output.write("print('Training started')\n")
output.write("model.fit(x_train, y_train, epochs={}, verbose=1, validation_data=(x_test, y_test))".format(numepochs))
output.write("test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)\n")
output.write("print(test_acc)\n")
output.write("log.write('Accuracy = {:.4f}\\n'.format(acc))\n")
output.write("log.close()\n")
output.close()
