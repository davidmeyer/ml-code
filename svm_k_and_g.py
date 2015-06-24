#
#       svm_k_and_g.py
#
#       Visualize the margin created by a Support Vector Machine (SVM)
#	with different kernels. Also take a look at the effect of the
#	parameter gamma.
#
#	Support Vector Machines (svm) are also called "Large Margin
#	Classifers" as they try to find the decision boundary with
#	the largest margin (and hence greatest confidence). In the
#	plot the dotted lines represent the margin and the solid line
#	is the hyperplane decision boundary.
#
#	In addition, the use of kernels (aka "the kernel trick") avoids
#	the explicit mapping that is needed to get linear learning algorithms
#	to learn non-linear function or decision boundaries. As a result
#	an SVM can learn non-linear features. See (among many others)
#	http://scikit-learn.org/stable/modules/svm.html#svm-mathematical-formulation
#	for a brief description of SVM mathematics.
#
#       Uses http://scikit-learn.org/ libsvm support
#
#       On Ubuntu:
#
#       # apt-get install build-essential python-dev python-setuptools \
#                         python-numpy python-scipy libatlas-dev       \
#                         libatlas3gf-base python-matplotlib
#
#
#
#
#	David Meyer
#	dmm@1-4-5.net
#	Fri Aug  1 11:26:24 2014
#
#	$Header: /mnt/disk1/dmm/ai/code/dmm/RCS/svm_k_and_g.py,v 1.2 2014/08/08 18:42:27 dmm Exp $
#


#
#       Get what we need
#
import numpy as np
import matplotlib.pyplot as plt
from   sklearn import svm
import sys, getopt
#
#
#	globals
#
DEBUG = 0
#
#
#
def main(argv):
#
#       get defaults
#
        kernel        = 'linear'
	kernel_string = "Linear"
        gamma         = 2
#
#       getopt doesn't like argv[0]
#
        try:
            opts,args = getopt.getopt(argv[1:],"hk:g:", ["kernel=", "gamma="])
        except getopt.GetoptError:
                print argv[0], "[-h -g <gamma> -k <kernel>]"
                sys.exit(2)
        for opt, arg in opts:
                if opt in ("-h", "--help"):
                        print argv[0], "[-h -g <gamma> -k <kernel>]"
                        sys.exit()
                elif opt in ("-k", "--kernel"):
                        kernel = arg
			if (DEBUG > 1):
	                        print "using kernel = ", kernel
#
#	Try to make the plot a bit more readable
#
			if (kernel == 'linear'):
				kernel_string = "Linear"
			elif (kernel == 'rbf'):
				kernel_string = "Guassian RBF"
			elif (kernel == 'poly'):
				kernel_string = "Polynomial"
			elif (kernel == 'sigmoid'):
				kernel_string = "Sigmoid"
			else:
				print "Unknown kernel (%s). Kernel can be one of linear, poly, rbf or sigmoid." % (kernel)
				sys.exit(2)
                elif opt in ("-g", "--gamma"):
                        gamma = arg
			if (DEBUG > 1):
	                        print "using gamma = ", gamma


#
#	For plotting purposes...
#
	display_string = "%s Kernel with Gamma = %s" % (kernel_string, gamma)
#
#	in case this is of interest...
#
	if (DEBUG > 1):
		print "display_string is %s" % (display_string)

#
#	As usual, make up a data set (X) and targets (Y)
#      

        X = np.c_[(.4, -.7),
                  (-1.5, -1),
                  (-1.4, -.9),
                  (-1.3, -1.2),
                  (-1.1, -.2),
                  (-1.2, -.4),
                  (-.5, 1.2),
                  (-1.5, 2.1),
                  (1, 1),
                  (1.3, .8),
                  (1.2, .5),
                  (.2, -2),
                  (.5, -2.4),
                  (.2, -2.3),
                  (0, -2.7),
                  (1.3, 2.1)].T

        Y = [0] * 8 + [1] * 8

#
#       fit the model
#
#
#       See http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
#	for parameters for svm.SVC. Only look at the effect of kernel
#	and gamma choices on decision boundaries here... 
#
#
        clf = svm.SVC(kernel=kernel, gamma=float(gamma))
        clf.fit(X, Y)
#
#	plot the line, the points, and the nearest vectors to the plane
#

        plt.figure(display_string, figsize=(4, 3))
        plt.clf()
        plt.scatter(clf.support_vectors_[:, 0],			\
                    clf.support_vectors_[:, 1],			\
                    s=80,					\
                    facecolors='none',				\
                    zorder=10)
        plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)
        plt.axis('tight')
        x_min = -3
        x_max = 3
        y_min = -3
        y_max = 3

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z      = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

#
#	Put the result into a color plot
#

        Z = Z.reshape(XX.shape)
        plt.figure(display_string, figsize=(4, 3))
	plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
	plt.contour(XX,					\
		    YY,					\
		    Z,					\
		    colors=['k', 'k', 'k'],		\
		    linestyles=['--', '-', '--'],	\
		    levels=[-.5, 0, .5])		

	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	
	plt.xticks(())
	plt.yticks(())
	plt.show()
#
#       call main to get it all going
#
if __name__ == "__main__":
        main(sys.argv)
