#
#	(very) basic autoencoder to recognize MNIST digits
#
#       interestingly, messing with 
#
#               training_epochs
#               training_batch_size
#
#       gives insight into how training proceess (training_batch_size around one not surprisingly yields random noise, ...)
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
DEBUG               = 2
learning_rate       = 0.01
train_batch_size    = 64
test_batch_size     = 256
display_step        = 1
training_epochs     = 5
training_batch_size = 2
#
#	MNIST parameters
#
img_size      = 28			# images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size	# tuple with height and width of images used to reshape arrays.
img_shape     = (img_size, img_size)	# number of colour channels for the images: 1 channel for gray-scale.
num_channels  = 1			# number of classes, one class for each of 10 digits.
# 
#	Network Parameters
#
n_input          = img_size_flat	# MNIST data input (img shape: 28*28)
n_hidden         = int(n_input/3)	# rule of thumb, but...
num_classes      = 10                   # not really used, as we're trying to reconstruct the image input
#
#
#
#	get MNIST data set and place holder for feed_dict
#
data  = input_data.read_data_sets("/tmp/data/", one_hot=True)
X     = tf.placeholder("float", [None, n_input])
#
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
#       really using these)
#
cls_true = data.test.cls[0:9]
#
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
encoder_op = encoder(X,tf.nn.sigmoid)
decoder_op = decoder(encoder_op,tf.nn.sigmoid)
#
#	decoder_op is our predicted value (y_pred)
#
y_pred = decoder_op
#
#	y_true is the imput X 
#
y_true = X
#
#	cost is just MSE (add regularzers next)
#
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#
#	use the Adam optimizer
#
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#
#       might try others...
#
#       optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
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
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()   


#
#       break up into batches and run the optimizer
#
def optimize(training_epochs,training_batch_size):
        start_time = time.time()
        for epoch in range(training_epochs):
            for i in range(training_batch_size):                                        # Loop over all batches
                batch_xs, _ = data.train.next_batch(train_batch_size)
                _, c = session.run([optimizer, cost], feed_dict={X: batch_xs})
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(c))
        end_time = time.time()
        time_dif = end_time - start_time
        print('\nOptimization Finished...(training_epochs: {:d},training_batch_size: {:d}, elapsed time: {:s})'
              .format(training_epochs,training_batch_size,str(timedelta(seconds=int(round(time_dif))))))



#
#
#       display_reconstruction
#
#       Run the encoder/decoder on the test set, compare originals to reconstruction
#
#       Compare original images with their test set reconstructions
#
def display_reconstruction(examples_to_show,fontsize):
        reconstruction = session.run(y_pred, feed_dict={X: data.test.images[:examples_to_show]})
        fig, axes = plt.subplots(2, 10, figsize=(10, 3))
        axes[0][0].set_title('MNIST', fontsize=fontsize)
        axes[1][0].set_title('Reconstruction', fontsize=fontsize)
        for i in range(examples_to_show):
                axes[0][i].set_xticks([])                   # has to be a better way to do this
                axes[0][i].set_yticks([])                   # ...
                axes[1][i].set_xticks([])                   # ...
                axes[1][i].set_yticks([])                   # still removing ticks
                axes[0][i].imshow(np.reshape(data.test.images[i], (28, 28)))
                axes[1][i].imshow(np.reshape(reconstruction[i], (28, 28)))
        fig.show()
        plt.draw()
        plt.waitforbuttonpress()                        # friendly for notebooks

#
#
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
#
#	visualize a few images with ground truth labels
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
training_epochs     = 10
training_batch_size = 64                               # minibatch size (essentially random to start)
#
optimize(training_epochs=training_epochs,training_batch_size=training_batch_size)
display_reconstruction(examples_to_show=10, fontsize=18)

