#
#       mnist_conv_net.py
#
#       Tensorflow convolutional network with dropout
#
#       See http://cs231n.github.io/convolutional-networks/ for a
#       reasonable intro to conv nets.
#
#       https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf is
#       the canonical reference for dropout.
#
#
#       Use Yann LeCun's MNIST database of handwritten digits
#       (http://yann.lecun.com/exdb/mnist/) 
#
#
#       NB: Looks like our training accuracy is only 0.949219, suggesting
#       that something is wrong (SOTA is > 99.7%)
#
#       See http://www.tensorflow.org/tutorials/mnist/pros/
#
#	David Meyer
#	dmm@1-4-5.net
#	Mon Nov 30 10:30:47 2015
#
#	$Header: $
#
#
#
#       imports
#
import input_data                       # put input_data.py in same dir
import tensorflow as tf                 
#
#       parameters
#
DEBUG               = 1                 # 1 -- see training progress
learning_rate       = 0.001
training_iterations = 100000
batch_size          = 128
display_step        = 10
#
#       get the mnist data set
#
mnist               = input_data.read_data_sets("/tmp/mnist/", one_hot=True)
#
#       network parameters
#
n_input             = 784               # MNIST data input (img shape: 28*28)
n_classes           = 10                # MNIST total classes (0-9 digits)
dropout             = 0.75              # Dropout, probability to keep units

#
#       build the tf graph
#
x                   = tf.placeholder(tf.types.float32, [None, n_input])
y                   = tf.placeholder(tf.types.float32, [None, n_classes])
keep_probability    = tf.placeholder(tf.types.float32)            
#
#       build model
#
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w,       \
           strides=[1, 1, 1, 1], padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1],              \
           strides=[1, k, k, 1], padding='SAME')

#
#       _foo is a parameter name for the conv_net function
#
#
def conv_net(_X, _weights, _biases, _dropout):
#
#
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])
#
#       1st convolution layer
#
#       alternate with max pooling, then apply dropout
#
#       The convolution has sride 1 and zero-padding so that the
#       28x28x1 input image is padded  32x32x1 (so the input
#       is the same size as the output)
#
#       So....apply 5x5x32 convolution (wc1) to get 28x28x32
#
#       'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]))
#
#       --> convolutional will compute 32 features for each 5x5 patch.
#       Its weight tensor will have a shape of [5, 5, 1, 32]. The first
#       two dimensions are the patch size, the next is the number of
#       input channels, and the last is the number of output channels.
#       We also have a bias vector (bc1) with a component for each
#       output channel.
#
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
#
#       apply max-pooling down-sample to 14x14x32
#
    conv1 = max_pool(conv1, k=2)
#
#       apply dropout
#
    conv1 = tf.nn.dropout(conv1, _dropout)
#
#       conv1 is now a complete convolutional/max pooling
#       plus dropout
#
#       Do the same for the 2nd convolution layer
#
#       zero-padding the 14x14x32 to 18x18x32
#       apply 5x5x32x64 convolution to get 14x14x64
#       max-pooling down to 7x7x64

    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    conv2 = max_pool(conv2, k=2)
    conv2 = tf.nn.dropout(conv2, _dropout)
#
#       Now do the fully connected layer
#
#       First, reshape the last conv layer (conv2) to fit output layer
#
    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
#
#       activation for the output layer is ReLU
#
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
#
#       want dropout here too
#
    dense1 = tf.nn.dropout(dense1, _dropout)
#
#       now predict and return the output class
#
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out
#
#
#       data structures for weights and biases
#
#       wc1: 5x5 conv, 1 input, 32 outputs
#       wc2: 5x5 conv, 32 inputs, 64 outputs
#       wd1: fully connected, 7*7*64 inputs, 1024 outputs
#       out: 1024 inputs, 10 outputs (class prediction)
#
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), 
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])), 
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])), 
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
#
#
#       build the actual model
#
#       y_hat is our predicted class
#
#
y_hat = conv_net(x, weights, biases, keep_probability)
#
#       cost and optimization
#
cost      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_hat, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
#       how well did we do?
#
#       tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1)) gives us a list
#       of booleans. To determine what fraction are correct, we cast
#       to floating point numbers and then take the mean.
#
#       For example, [True, False, True, True] would become [1,0,1,1],
#       which with
#
#       tf.reduce_mean(tf.cast(correct_prediction, tf.types.float32))
#
#       would become 0.75 (accuracy).
#
#       See http://www.tensorflow.org/tutorials/mnist/pros
#
#
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.types.float32))
#
#       Now get it all going
#
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 1
#
#       Keep training until max iterations
#
    while ((step * batch_size) < training_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#
#       Fit training using batch data
#
        sess.run(optimizer,                                             \
                 feed_dict={x: batch_xs, y: batch_ys,                   \
                            keep_probability: dropout})
        if step % display_step == 0:
#
#       Calculate batch accuracy
#
            acc = sess.run(accuracy,
                           feed_dict={x: batch_xs, y: batch_ys,         \
                                      keep_probability: 1.0})
#
#       Calculate batch loss
#
            loss = sess.run(cost,
                        feed_dict={x: batch_xs, y: batch_ys,            \
                                   keep_probability: 1.0})
            if (DEBUG): 
                 print "Iteration " + str(step*batch_size) +            \
                       ", Minibatch Loss = " + "{:.6f}".format(loss) +  \
                       ", Training Accuracy = " + "{:.5f}".format(acc)
        step += 1
    print "done..."

#
#       now calculate accuracy
#
    print "Accuracy: ",                                                 \
          sess.run(accuracy,                                            \
                   feed_dict={x: mnist.test.images[:256],               \
                              y: mnist.test.labels[:256],               \
                              keep_probability: 1.0})
