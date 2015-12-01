#
#       pca_kmeans_ni.py
#
#       See if PCA/K-means network_intrusion_detection.csv makes any
#       sense. The answer is kinda.
#
#       Requires: sklearn, numpy, pandas, and matplotlib
#
#	David Meyer
#	dmm@1-4-5.net
#	Tue Apr 7 12:15:37 2015
#
#	$Header: /mnt/disk1/dmm/ai/code/dmm/RCS/pca_kmeans_ni.py,v 1.3 2015/04/08 15:34:21 dmm Exp $
#

#
import matplotlib.pyplot     as plt
import pandas                as pd              # use pandas for reading csvs
import numpy                 as np
import sklearn.preprocessing as pp
from   sklearn               import metrics
from   sklearn.cluster       import KMeans
from   sklearn.decomposition import PCA
#
#
#       Globals, if any
#
#
DEBUG = 0
DB    = 'network_intrusion_detection.csv'
#
#
#       standard stuff
#
#
plt.style.use('ggplot')
np.random.seed(42)
#
#
#       can get the data set this way
#
# # pip install wget (if you have pip)
# import wget
# url = 'https://azuremlsampleexperiments.blob.core.windows.net/datasets/network_intrusion_detection.csv'
# csv = wget.download(url)
# df  = pd.read_csv(csv)
#
#
#       I have it locally, so just read the csv
#
df           = pd.read_csv(DB)
#
#       delete non-numeric columns (could code these...)
#
df           = df.drop(['protocol_type','service','flag','class'],1)
#
#       normalize
#
data         = pp.scale(df)
#
#       reduce to 2-D for viz
#
reduced_data = PCA(n_components=2).fit_transform(data)
#
#       run kmeans with 8 clusters
#
kmeans       = KMeans(init='k-means++', n_clusters=8, n_init=10)
kmeans.fit(reduced_data)
#
#       Step size of the mesh for viz. Decrease to increase the
#       quality of the viz
#
h            = .02         
#
#       Plot the decision boundary. For that, assign a color to each space
#       Note: this code is only good for 2-D viz...
#
#
#       Get min/max for the columns
#
x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
#
#       put it all together
#
xx, yy       = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
#
#       get labels for each point in mesh using last trained model
#
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#
#       put the result into a color plot
#
Z = Z.reshape(xx.shape)
#
#
#       standard plot of the 2-D projected output
#
#
plt.figure(1)
plt.clf()
plt.imshow(Z,
           interpolation = 'nearest',
           extent        = (xx.min(), xx.max(), yy.min(), yy.max()),
           cmap          = plt.cm.Paired,
           aspect        = 'auto',
           origin        = 'lower')
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
#
#       plot the centroids as white crosses
#
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0],
            centroids[:, 1],
            marker     = 'x',
            color      = 'w',
            s          = 169,
            linewidths = 3,
            zorder     = 10)
#
#       get a pretty title
#
plt.title("K-means clustering of {0} (PCA-reduced)\nCentroids are marked with white cross".format(DB))
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
