import random
import sys

if len(sys.argv) != 2:
    print("Usage: python genn.py <graph file name>")
    quit()

mnist_width = 28
mnist_height = 28
nnlayers = 3
nnwidth = 512
numclasses = 10
batch_size = 100

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
output = open(out_fname, "w")

output.write("import tensorflow as tf\n")
output.write("import random\n")
output.write("\n")
output.write("mnist = tf.keras.datasets.mnist\n")
output.write("(x_train, y_train), (x_test, y_test) = mnist.load_data()\n")
output.write("x_train = tf.cast(x_train, tf.float32)\n")
output.write("x_test = tf.cast(x_test, tf.float32)\n")
output.write("\n")
output.write("x_train, x_test = x_train / 255.0, x_test / 255.0\n")
output.write("\n")
output.write("num_classes = {}\n".format(numclasses))
output.write("\n")
output.write("x_train = tf.reshape(x_train, [-1, {}])\n".format(mnist_width*mnist_height))
output.write("x_test = tf.reshape(x_test, [-1, {}])\n".format(mnist_width*mnist_height))
output.write("\n")
output.write("y_train = tf.keras.utils.to_categorical(y_train, num_classes)\n")
output.write("y_test = tf.keras.utils.to_categorical(y_test, num_classes)\n")
output.write("\n")
output.write("print(x_train.shape)\n")
output.write("\n")
output.write("learning_rate = 0.001\n")
output.write("num_epochs = 20\n")
output.write("batch_size = {}\n".format(batch_size))
output.write("\n")
output.write("xavier = tf.keras.initializers.GlorotUniform()\n")
output.write("\n")

output.write("### num_inputs, num_outputs, num_hiddennodes, num_edges\n")
output.write("### {} {} {} {}\n".format(num_inputs, num_outputs, num_hiddennodes, num_edges))

output.write("W = tf.Variable(xavier([{}]))\n".format(num_edges))
output.write("B = tf.Variable(tf.random.normal([{}]))\n".format(num_nodes))

for i in range(num_inputs + num_outputs, num_nodes):
    output.write("@tf.function\n")
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
    output.write("    return(result)\n\n")


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

output.write("\n")
output.write("@tf.function\n")
output.write("def Cost(X, Y):\n")
output.write("    return(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Hypothesis(X), labels=Y)))\n")
output.write("\n")
output.write("def Minimize(X, Y):\n")
output.write("    loss = lambda: Cost(X ,Y)\n")
output.write("    tf.keras.optimizers.Adam(learning_rate).minimize(loss, [W, B])\n")
output.write("\n")
output.write("def CorrectPrediction(X, Y):\n")
output.write("    return(tf.equal(tf.argmax(Hypothesis(X), axis=1), tf.argmax(Y, axis=1)))\n")
output.write("\n")
output.write("def Accuracy(X, Y):\n")
output.write("    return(tf.reduce_mean(tf.cast(CorrectPrediction(X, Y), tf.float32)))\n")
output.write("\n")
###output.write("def total_accuracy(X, Y):\n")
###output.write("    acc_sum = 0\n")
###output.write("    for i in range(len(X)):\n")
###output.write("        x_, y_ = X[i], Y[i]\n")
###output.write("        acc = Accuracy(x_, y_)\n")
###output.write("        acc_sum += acc\n")
###output.write("    return acc_sum/len(X)\n")
output.write("\n")
output.write("for epoch in range(num_epochs):\n")
output.write("    avg_cost = 0\n")
output.write("    num_batch = int(len(x_train) / batch_size)\n")
output.write("\n")
output.write("    start_batch, end_batch = 0, batch_size\n")
output.write("    cost_sum = 0\n")
output.write("    total_acc = 0\n")
output.write("    for i in range(num_batch):\n")
###output.write("    for i in range(len(x_train)):\n")
output.write("        batch_xs, batch_ys = x_train[start_batch:end_batch], y_train[start_batch:end_batch]\n")
###output.write("        x_, y_ = x_train[i], y_train[i]\n")
output.write("        Minimize(batch_xs, batch_ys)\n")
###output.write("        Minimize(x_, y_)\n")
output.write("        cost_val = Cost(batch_xs, batch_ys)\n")
output.write("        cost_sum += cost_val\n")
output.write("        start_batch = start_batch + batch_size\n")
output.write("        end_batch = end_batch + batch_size\n")
output.write("        acc = Accuracy(batch_xs, batch_ys)\n")
output.write("        total_acc += acc\n")
output.write("    print('Epoch: {:04d}, Cost: {:.9f}, Acc: {:.4f}'.format(epoch + 1, cost_sum, total_acc/num_batch))\n")
output.write("\n")
output.write("print('Learning finished')\n")
output.write("print('Accuracy = {:.4f}'.format(Accuracy(x_test, y_test)))\n")
