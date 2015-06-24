//
//	pca+km.scala
//
//	Very simple PCA/KMeans anomanly dectection in KDD Cup '99
//
//	Data set: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
//
//	See http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names for a
//	description of the features. 
//
//	To run cut and paste into the spark scala REPL
//
//	./bin/spark-shell
//
//	A typlical line looks like in the kddcup99 data set looks
//	like
//
//      0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,
//	  0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,
//        0.00,0.00,normal.
//
//	Open:
//
//	(i).	Is there a good way to shorten
//		org.apache.spark.mllib.linalg.Vector in function
//		definitions (see e.g., distance below)? 
//
//	(ii).	Build a standalone .jar with sbt
//
//
//
//
//	David Meyer
//	dmm@1-4-5.net
//	Fri Apr 24 08:00:30 2015
//
//	$Header: /mnt/disk1/dmm/ai/spark/dmm/km/RCS/pca+km.scala,v 1.4 2015/05/01 14:22:34 dmm Exp $
//

import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering._
import scala.collection.mutable.ArrayBuffer
import scala.math.sqrt
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.feature.StandardScaler

//
//	Various constants, etc
//

val DEBUG       = 0

val PRTS        = 1	// location of protocol in rawData
val SRVS        = 2	// location of service in rawData
val FLGS        = 3	// location of flags in rawData

//
//	Indicies
//
//	Subtract offset for use with encodings
//	(Array[scala.collection.immutable.Map[String,Int]])
//

val offset	= 1				// Array's start at 0
val PIDX	= PRTS - offset
val SIDX        = SRVS - offset
val FIDX	= FLGS - offset

//
//	PCA hyperparameters
//

val dimensions  = 30

//
//	KMeans hyperparameters
//

val K	       = 45
val runs       = 20
val epsilon    = 1.0e-6

//
//	Get the dataset
//
//	Can't do much more than 10000 data points on my laptop
//
//	% zcat kddcup.data.gz | head -10000 >! kddcup10000.data
//

val kddcupData = "kddcup10000.data"
val datafile   = "%s/%s".format(System.getProperty("user.dir"), kddcupData)
val rawData    = if (new java.io.File(datafile).exists) {
                     sc.textFile(datafile)
                 } else {
                     println("Datafile %s not found.\n".format(datafile))
                     sys.exit(1)
                 }

//
//	ETL the dataset
//
//	Encode non-numeric fields (protocol_type, services, and flags)
//
//	Build tuples of the form (<non-numeric>, encoding) where
//	the encoding is just the zip'ed index, .e.g, 
//
//	scala> protocols
//	res2: Array[(String, Int)] = Array((icmp,0), (udp,1),(tcp,2))
//
//
//	Note: this doesn't really make a lot of sense since there
//	isn't a good distance metric for "categoricals".
//
var protocols = rawData.map(_.split(",")(PRTS)).distinct.collect.zipWithIndex
var services  = rawData.map(_.split(",")(SRVS)).distinct.collect.zipWithIndex
var flags     = rawData.map(_.split(",")(FLGS)).distinct.collect.zipWithIndex

//
//	Build an array of maps for protocols, services, and flags
//	so that encodings(xIDX)("string") returns the encoding of
//	that string under xIDX.
//

val encodings = Array(protocols.toList.toMap,	// encodings(PIDX)
                       services.toList.toMap,	// encodings(SIDX)
                          flags.toList.toMap)	// encodings(FIDX)

//
//
//	Build a vector from the line by spliting on ",",
//	overwriting the symbolic name with their encoding (String)
//	
//	buffer(PRTS) has the protocol
//	buffer(SRVS) has the service
//	buffer(FLGS) has the flags
//
//	Turn the whole thing into a vector, then build tuples
//	(vector, label)
//
//

val dataAndLabel = rawData.map {line =>
  val buffer     = ArrayBuffer[String]()
  buffer.appendAll(line.split(","))
  buffer(PRTS)   = encodings(PIDX)(buffer(PRTS)).toString
  buffer(SRVS)   = encodings(SIDX)(buffer(SRVS)).toString
  buffer(FLGS)   = encodings(FIDX)(buffer(FLGS)).toString
  val label      = buffer.remove(buffer.length-1)
  val vector     = Vectors.dense(buffer.map(_.toDouble).toArray)
  (vector,label)
}

val data = dataAndLabel.map(_._1).cache()

//
//	Reduce to 'dimensions' dimensions with PCA
//

val mat: RowMatrix = new RowMatrix(data)
val pc:  Matrix    = mat.computePrincipalComponents(dimensions)
val rdata          = mat.multiply(pc).rows
val rdataAndLabel  = rdata.zip(dataAndLabel.map(_._2))

//
//	normalize
//
//      Standardizes features by making the mean (about) zero and
//	scaling to unit variance using column summary
//
//	Get the standard scaler, fit to the data
//

val scaler = new StandardScaler(true, true).fit(rdata)

//
//	transform RDD into new RDD with normalized data
//

val normalizedData = rdata.map {vector =>
    val normalized = scaler.transform(vector)
    (normalized)
}.cache()

//
//	see what we learned
//

val kmeans = new KMeans()
kmeans.setK(K)
kmeans.setRuns(runs)
kmeans.setEpsilon(epsilon)
val model = kmeans.run(normalizedData)
val clusterAndLabel = rdataAndLabel.map {
     case (normalizedData,label) => (model.predict(normalizedData), label)}
val clusterLabelCount = clusterAndLabel.countByValue
clusterLabelCount.toList.sorted.foreach {
     case ((cluster,label), count) => println(f"$cluster%1s$label%18s$count%8s")}

