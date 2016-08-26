#
#	(very) basic autoencoder to recognize MNIST digits
#
#       interestingly, messing with 
#
#               training_epochs
#               training_batch_size
#
#       gives insight into how training proceess (training_batch_size
#       around 1 not surprisingly yields random noise, ...)
#
#
#	David Meyer
#	dmm@1-4-5.net
#	Tue Aug 23 13:35:00 2016
#
#	$Header: $
#
#
from   __future__                          import division, print_function, absolute_import
from   tensorflow.examples.tutorials.mnist import input_data
from   sklearn.metrics                     import confusion_matrix
from   datetime                            import timedelta
import tensorflow                          as     tf
import numpy                               as     np
import matplotlib.pyplot                   as     plt
import time
import math
#
#	global parameters
#
DEBUG               = 1                         # more debug
USE_REGULARIZER     = 1                         # use regularization?
learning_rate       = 0.01
test_batch_size     = 256
display_step        = 1
#
#	MNIST parameters
#
img_size            = 28			# images 28 x 28
img_size_flat       = img_size * img_size	# flattened
img_shape           = (img_size, img_size)	# shape
num_channels        = 1         		# 1 is greyscale
# 
#	Network Parameters
#
n_input             = img_size_flat             # input size (img shape: 28*28)
#
#       rule of thumb: n_hidden = n_input/3
#
#n_hidden            = int(n_input/3) 
n_hidden            = 100                       # testing
num_classes         = 10                        # not used
#
#
#	get MNIST data set and place holder for feed_dict
#
data  = input_data.read_data_sets("/tmp/data/", one_hot=True)
X     = tf.placeholder("float", [None, n_input])
#
#	one-hot encoded class labels [0-9, one hot encoded]
#
data.test.cls = np.argmax(data.test.labels, axis=1)
#
#	get the images
#
images = data.test.images[0:9]
#
#	Get the true classes for those images (again, autoencoder, not
#       really using these, execpt for checking that we loaded MNIST
#       correctly (see plot_images below)
#
cls_true = data.test.cls[0:9]
#
#	weights and biases
#
#       Note: one hidden layer. Thi sapproach is cool as it is easily
#       generalized to many hidden layers.
#
#
weights = {
    'encoder': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'decoder': tf.Variable(tf.random_normal([n_hidden, n_input]))
}
biases = {
    'encoder': tf.Variable(tf.random_normal([n_hidden])),
    'decoder': tf.Variable(tf.random_normal([n_input])),
}
#
#	encoder/decoder
#	
#       try tf.nn.sigmoid, tf.nn.relu, etc for nonlinearity
#       nonlinearity=False means transfer function (aka activation funtion)
#       g(x) = x
#
def encoder(x, nonlinearity=False):
    code = tf.add(tf.matmul(x, weights['encoder']), biases['encoder'])
    if nonlinearity:
        code = nonlinearity(code)
    return code

def decoder(code, nonlinearity=False):
    reconstruction = tf.add(tf.matmul(code, weights['decoder']), biases['decoder'])
    if nonlinearity:
        reconstruction = nonlinearity(reconstruction)
    return reconstruction
#
#	get the encoding and decoding operations
#
#       relu seems less efficient here
#
#
#       first encode
#
encoder_op = encoder(X,tf.nn.sigmoid)
#
#       then decode
#
decoder_op = decoder(encoder_op,tf.nn.sigmoid)
#
#	decoder_op is our predicted value (y_pred)
#
y_pred = decoder_op
#
#	y_true is the input X 
#
y_true = X
#
#
#       with regularization
#
#       cost = tf.add(tf.reduce_mean(tf.pow(y_true - y_pred, 2)),
#                     tf.mul(reg_constant,tf.reduce_sum(reg_losses)))
#
# Epoch: 0001 cost = 0.455753744
# Optimization Finished...(training_epochs: 1,training_batch_size: 1, elapsed time: 0:00:00)
# Epoch: 0001 cost = 0.189976141
# Optimization Finished...(training_epochs: 1,training_batch_size: 9, elapsed time: 0:00:00)
# Epoch: 0001 cost = 0.071388490
# Optimization Finished...(training_epochs: 1,training_batch_size: 90, elapsed time: 0:00:02)
# Epoch: 0001 cost = 0.028894797
# Optimization Finished...(training_epochs: 1,training_batch_size: 900, elapsed time: 0:00:30)
#       w/o regularization
#
#       cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#
# Epoch: 0001 cost = 0.431725562
# Optimization Finished...(training_epochs: 1,training_batch_size: 1, elapsed time: 0:00:00)
# Epoch: 0001 cost = 0.190942019
# Optimization Finished...(training_epochs: 1,training_batch_size: 9, elapsed time: 0:00:00)
# Epoch: 0001 cost = 0.072323456
# Optimization Finished...(training_epochs: 1,training_batch_size: 90, elapsed time: 0:00:02)
# Epoch: 0001 cost = 0.034361549
# Optimization Finished...(training_epochs: 1,training_batch_size: 900, elapsed time: 0:00:30)
#
#
reg_losses   = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0.01 
#
#
if (USE_REGULARIZER):
        error = tf.add(tf.reduce_mean(tf.square(tf.sub(y_true,y_pred))),
                       tf.mul(reg_constant,tf.reduce_sum(reg_losses)))
else:
        error = tf.reduce_mean(tf.square(tf.sub(y_true,y_pred)))

#
#	use the Adam optimizer
#
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(error)
#
#       might try others, e.g., 
#
#       optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(error)
#
#
#	Get tensorlfow going
#
init = tf.initialize_all_variables()
#
#	Launch the graph (use InteractiveSession as that is more convenient while using Notebooks)
#
session = tf.InteractiveSession()
session.run(init)
#
#	check that we loaded MNIST correctly
#
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, ax = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(ax.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "Label: {0}".format(cls_true[i])
        else:
            xlabel = "Label: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])               # get rid of ticks
        ax.set_yticks([])
    plt.show()   
#
#       break up into batches and run the optimizer
#
def optimize(training_epochs,training_batch_size):
        start_time = time.time()
        for epoch in range(training_epochs):
            for i in range(training_batch_size):   # Loop over all batches
                batch_xs, _ = data.train.next_batch(training_batch_size)
                _, c = session.run([optimizer, error], feed_dict={X: batch_xs})
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "error =", "{:.9f}".format(c))
        end_time = time.time()
        time_dif = end_time - start_time
        print('Optimization Finished...training_epochs: {:d},training_batch_size: {:d}, elapsed time: {:s}'
              .format(training_epochs,training_batch_size,str(timedelta(seconds=int(round(time_dif))))))

#
#
#       display_reconstruction
#
#       Run the encoder/decoder on the test set, compare originals to reconstruction
#
#       Compare original images with their test set reconstructions
#
#
def display_reconstruction(examples_to_show,fontsize):
        reconstruction = session.run(y_pred,
                                     feed_dict={X: data.test.images[:examples_to_show]})
        fig, ax = plt.subplots(2,10,figsize=(10,3))
        aboutmiddle = int((examples_to_show/2) - 1)                     # sort of
        ax[0][aboutmiddle].set_title('MNIST',                           # move over then center
                           horizontalalignment='center',
                           fontsize=fontsize)  
        ax[1][aboutmiddle].set_title('Reconstruction',                  # move over then center
                           horizontalalignment='center',
                           fontsize=fontsize)
        for i in range(examples_to_show):
            ax[0][i].set_xticks([])                                     # has to be a better way
            ax[0][i].set_yticks([])                                     # ...
            ax[1][i].set_xticks([])                                     # ...
            ax[1][i].set_yticks([])                                     # still removing ticks
            ax[0][i].imshow(np.reshape(data.test.images[i],(28, 28)))
            ax[1][i].imshow(np.reshape(reconstruction[i],  (28, 28)))
        fig.show()
        plt.draw()

#       plt.waitforbuttonpress()                      # friendly for notebooks
#
#
#	Check out the data set (if DEBUG)
#
if (DEBUG):
        print("\nMNIST data sets (set DEBUG=1 to visualize):")
        print("- Training-set:\t\t{}".format(len(data.train.labels)))
        print("- Test-set:\t\t{}".format(len(data.test.labels)))
        print("- Validation-set:\t{}\n".format(len(data.validation.labels)))
#
#	visualize a few images with ground truth labels
#
        if (DEBUG > 1): 
                plot_images(images=images, cls_true=cls_true)

#
#       start with training_epochs = 1, training_batch_size = 1
#
#       Adjust these for interesting results, e.g.,
#
#       training_epochs     = 1
#       training_batch_size = 1
#
#
training_epochs = 10                            # arbitrary
#
#
#       doesn't quite display right, but ...
#
for batch_size in [1, 9, 90, 1000]:              # 1+9 = 10, 10+90 = 100, ...= 1000
        optimize(training_epochs=training_epochs,training_batch_size=batch_size)
        display_reconstruction(examples_to_show=10,fontsize=18)
#
# optimize(training_epochs=training_epochs,training_batch_size=training_batch_size)
# display_reconstruction(examples_to_show=10, fontsize=18)

