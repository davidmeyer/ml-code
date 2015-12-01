#
#       mnist_softmax_regression.py
#
#       softmax regression using TensorFlow. Uses Uses MNIST
#       database of handwritten digits (http://yann.lecun.com/exdb/mnist/) 
#
#       See http://www.tensorflow.org/tutorials/mnist/beginners
#
#	David Meyer
#	dmm@1-4-5.net
#	Mon Nov 30 07:44:52 2015
#
#	$Header: $
#
#
#
#       Need these
#
import input_data               # google code to import MNIST data sets
import tensorflow as tf
#
#       parameters
#
DEBUG           = 1             # set DEBUG = 1 to watch optimization logs
learning_rate   = 0.01
training_epochs = 25
batch_size      = 100
display_step    = 1
#
#       get MNIST data, one hot encoded, write it to /tmp/data
#
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#
#
#       build tf computation graph
#
#       MNIST data image has shape 28*28=784
#
x = tf.placeholder("float", [None, 784])
#
#       0-9 digits recognition => 10 classes (one hot)
#
y = tf.placeholder("float", [None, 10]) 
#
#       Model parameters
#
W = tf.Variable(tf.zeros([784, 10]), name="weight_matrix")
b = tf.Variable(tf.zeros([10]),      name="bias_vector")
#
#       model (softmax)
#
y_hat = tf.nn.softmax(tf.matmul(x, W) + b)
#
#       use the (convex) cross entropy error cost
#
#       special case of cross entropy where y = 1, where
#
#       L(W) = - \frac{1}{N}\sum\limits_{n = 1|^{N} H(p_n,q_n)
#            = - \frac{1}{N}\sum\limits_{n = 1|^{N}
#               [y_{n}\log \hat{y}_{n} + ( 1 - y_{n}) \log (1 - \hat{y}_{n}]
#       where
#       \hat{y}_{n} \equiv g(w \cdot x_{n}) and g(z) is the logistic function
#
#
cross_entropy = -tf.reduce_sum(y*tf.log(y_hat))
#
#       Train with GD/minibatch
#
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
#
#       Initialize everythying
# 
init = tf.initialize_all_variables()
#
#       Run it all
# 
with tf.Session() as sess:
    sess.run(init)
#
#       Training cycle
#
    for epoch in range(training_epochs):
        avg_cost    = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
#
#       Loop over all batches
#
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#
#       Fit training using batch data
#
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
#
#       Compute average loss
#
            avg_cost += sess.run(cross_entropy,                 \
                             feed_dict={x: batch_xs, y: batch_ys})/total_batch
#
#       Display logs per epoch step
#
        if (DEBUG):
                if (epoch % display_step) == 0:
                   print "Epoch:", '%04d' % (epoch+1),          \
                   "cost=", "{:.9f}".format(avg_cost)

    print "Done"

#
#       
#       Test model
#
#       Notes: tf.argmax is an extremely useful function which gives you
#       the index of the highest entry in a tensor along some axis. For
#       example, tf.argmax(y,1) is the label our model thinks is most
#       likely for each input, while tf.argmax(y_,1) is the correct label.
#       We can use tf.equal to check if our prediction matches the truth.
#
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
#
#
#       At this point correct_prediction is
#
#       Tensor("Equal:0", shape=TensorShape([Dimension(None)]), dtype=bool)
#
#       i.e., correct_prediction is a list of booleans. To determine what
#       fraction are correct, we cast to floating point numbers and then
#       take the mean. For example, [True, False, True, True] would become
#       [1,0,1,1] which would become 0.75.
#
#
#
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
#       accuracy:  Tensor("Mean:0", shape=TensorShape([]), dtype=float32)
#
#
    print "Accuracy:",                                                  \
          accuracy.eval({x: mnist.test.images, y: mnist.test.labels})


