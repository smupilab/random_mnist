import networkx as nx
import random
import sys
import queue

if len(sys.argv) != 2:
    print("Usage: python genn.py <graph pickle file name>")
    quit()

batch_size = 100
numepochs = 20

graph_fname = sys.argv[1]
G = nx.read_gpickle(graph_fname)

num_inputs = 28 * 28
num_outputs = 10
num_hiddennodes = G.number_of_nodes() - num_inputs - num_outputs
num_edges = G.number_of_edges()
num_nodes = num_inputs + num_outputs + num_hiddennodes

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
    output.write("X_{} = tf.gather(flat_input, [{}], axis=1)\n".format(i, i))
output.write("\n")

for n in nx.topological_sort(G):
    if (n[0] != "X"):
        output.write("{} = tf.keras.layers.Dense(1, activation='relu')(tf.concat([".format(n))
        # edge u -> v
        E = list(G.in_edges(n))
        for u in range(len(E)-1):
            output.write("{}, ".format(E[u][0]))
        output.write("{}], 1))\n".format(E[-1][0]))

# final layer
output.write("final_nodes = tf.concat([")
for o in range(9):
    output.write("O_{}, ".format(o))
output.write("O_9], 1)\n")
output.write("\n")

output.write("final_output = tf.nn.softmax(final_nodes, axis=1)\n")
output.write("\n")
output.write("model = tf.keras.Model(model_input, final_output)\n")
output.write("model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n")
output.write("## model.summary()\n")
output.write("print('Training started')\n")
output.write("model.fit(x_train, y_train, epochs={}, verbose=1, validation_data=(x_test, y_test))\n".format(numepochs))
output.write("test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)\n")
output.write("print(test_acc)\n")
output.write("log.write('Accuracy = {}\\n'.format(test_acc))\n")
output.write("log.close()\n")
output.close()
