#
#
#	cffnb.py
#
#       Classification with Feedfoward Neural Network using Backpropagation
#
#	Build a network with two hidden layers with sigmoid neurons and
#	softmax neurons at the output layer. Train with backpropagation.
#
#	Make up training, cross-validation and test data sets if you don't
#	have some that you like (or otherwise).
#
#	Visualize the decision boundary as it evolves...
#
#	Requires:
#
#	python-numpy python-scipy python-matplotlib ipython
#	ipython-notebook python-pandas python-sympy python-nose
#
#	then (at least on Ubuntu):
#
#	sudo apt-get install python-dev python-scipy python-pip
#	sudo pip install pybrain
#
#	David Meyer
#	dmm@1-4-5.net
#	Thu Jul 31 09:54:22 2014
#
#	$Header: /mnt/disk1/dmm/ai/code/dmm/RCS/cffnb.py,v 1.8 2014/08/01 14:45:30 dmm Exp $
#


#
#       Get what we can from PyBrain (http://www.pybrain.org)
#
#       Note http://http://scikit-learn.org/ is a reasonable alternative
#       for non-ANN learning
#
#
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure		 import FeedForwardNetwork,FullConnection
from pybrain.structure		 import LinearLayer, SigmoidLayer
#
#
#       Need pylab if we want to do graphical output
#
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
#
#	Globals
#
DEBUG = 0
#
#
#       define a few functions for later use
#
#
#
#       pretty print data sets
#
def pp(trndata):
        length = len(trndata)
        print "Number of training patterns: ",length
        print "Input and output dimensions: ",trndata.indim,trndata.outdim
        print "input, target, class:"
        x = 0
        while (x < length):
                print trndata['input'][x],      \
                      trndata['target'][x],     \
                      trndata['class'][x]
                x += 1


#
#       build synthetic training data set(s) if you like...
#
means   = [(-1,0),(2,4),(3,1)]
cov     = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
alldata = ClassificationDataSet(2, 1, nb_classes=3)
for n in xrange(400):
    for klass in range(3):
        input = multivariate_normal(means[klass],cov[klass])
        alldata.addSample(input,[klass])

#
#       Randomly split the dataset into 75% training/25% test
#
tstdata, trndata = alldata.splitWithProportion(0.25)
#
#       Encode classes with one output neuron per class.
#       Note that this operation duplicates the original
#       targets and stores them in an (integer) field named
#       'class'. i.e., trndata['class']
#
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()
#
#       inspect dataset if you want...
#
if (DEBUG > 2):
	pp(trndata)

#
#	now build a feedforward neural network
#
#	Configuration:
#
#		Input Layer dimension: 2
#		2 hidden layers with 5 sigmoid neurons
#		Output layer has 3 Softmax neurons
#
net                = FeedForwardNetwork()
inLayer            = LinearLayer(2)
hiddenLayer1       = SigmoidLayer(5)
hiddenLayer2       = SigmoidLayer(5)
outLayer           = SoftmaxLayer(3)
#
#	add those layers (modules)
#
net.addInputModule(inLayer)
net.addModule(hiddenLayer1)
net.addModule(hiddenLayer2)
net.addOutputModule(outLayer)
#
#	do the plumbing
#
in_to_hidden1      = FullConnection(inLayer,hiddenLayer1)
hidden1_to_hidden2 = FullConnection(hiddenLayer1,hiddenLayer2)
hidden2_to_out     = FullConnection(hiddenLayer2,outLayer)
#
net.addConnection(in_to_hidden1)
net.addConnection(hidden1_to_hidden2)
net.addConnection(hidden2_to_out)
net.sortModules()
#
#	activate on the training data set
#
net.activateOnDataset(trndata)
#
#	build a backpropagation trainer
#
trainer = BackpropTrainer(net,			\
			  dataset=trndata,	\
			  momentum=0.1,		\
			  verbose=True,		\
			  weightdecay=0.01)

#
#	Generate a square grid of data points and put it into
#	a dataset, which we can then classify to get a nice
#	contour field for visualization...so the target values
#	for this data set aren't going to be used...
#
ticks    = arange(-3.,6.,0.2)
X,Y      = meshgrid(ticks, ticks)
#
#	Note need column vectors in dataset (not arrays)
#
griddata = ClassificationDataSet(2,1, nb_classes=3)
for i in xrange(X.size):
	griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
griddata._convertToOneOfMany()			# for the ffnn
for i in range(50):
	trainer.trainEpochs(1)			# one a a time for viz...
	trnresult = percentError(trainer.testOnClassData(), trndata['class'])
	tstresult = percentError(trainer.testOnClassData(dataset=tstdata ),	\
				 tstdata['class'] )
	print "epoch: %4d" % trainer.totalepochs,				\
                "  train error: %5.2f%%" % trnresult,				\
		"  test error: %5.2f%%" % tstresult

	out = net.activateOnDataset(griddata)
	out = out.argmax(axis=1)		# the highest output activation
        out = out.reshape(X.shape)
	figure(1)
	ioff()					# interactive graphics off
        clf()					# clear the plot
	hold(True)				# overplot on
	for c in [0,1,2]:
		here, _ = where(tstdata['class'] == c)
		plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
	if out.max() != out.min():		# safety check against flat field
		contourf(X, Y, out)		# plot the contour
	ion()					# interactive graphics on
	draw()					# update the plot


#
#	let this hang around until cntrl-C or whatever
#
ioff()
show()
