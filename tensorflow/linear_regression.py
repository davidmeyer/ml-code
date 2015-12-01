#
#
#	linear_regression.py
#
#       linear regreession in tensorflow
#
#
#	David Meyer
#	dmm@1-4-5.net
#	Tue Nov 24 13:41:24 2015
#
#

#
#	need this...
#
import tensorflow        as tf
import numpy             as np
import matplotlib.pyplot as plt
#
#	constants, etc
#
DEBUG = 0		# set DEBUG = 1 to see traing progress
N     = 100             # number of (generated) samples
rng   = np.random

#
#
#       make up some training data (-1 < x < 1, ....)
#
train_X = np.linspace(-1, 1, (N+1))           
#
#       create a y value which is approximately linear
#       but with some random noise (notiing that labels
#	are always noisy, cf mechanical turk)
#
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 
#
#       symbolic variables for computation graph (see later)
#
X = tf.placeholder("float")
Y = tf.placeholder("float")
#
#       linear regression is just y = W*X + b
#
def model(X, W, b):
        return tf.add(tf.mul(W, X), b)
#
#       create a shared variable (like theano.shared) for the weight, bias
#
W = tf.Variable(rng.randn(), name="weights") 
b = tf.Variable(rng.randn(), name="bias")
#
#       with that \hat{y} is y_hat = W*X + b
#
y_hat = model(X, W, b)
#
#       cost function is squared error
#
cost = (tf.pow(Y-y_hat, 2))
#
#       go fit the data with SGD, minimizing cost
#
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#
#       now, fire up the session
#
sess = tf.Session()
init = tf.initialize_all_variables() 
sess.run(init)
#
#       run it all
#
#	Note: set DEBUG = 1 if you want to see the progress of W and b
#
for i in range(N):
	for (x, y) in zip(train_X, train_Y):
		sess.run(train_op, feed_dict={X: x, Y: y})
	if (DEBUG):
		print(sess.run(W), sess.run(b)) 
#
#	plot results
#
#	pretty for the legend
#
legstr = "y = %f*X + %f" % (sess.run(W), sess.run(b))
#
#	plot it all
#
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label=legstr)
plt.legend(loc="upper left")
plt.show()
