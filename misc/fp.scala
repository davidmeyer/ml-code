//
//
//	fp.scala
//
//	FP Growth Algorithm
//
//
//	David Meyer
//	dmm@1-4-5.net
//	Tue May  5 11:37:39 2015
//
//	$Header: /mnt/disk1/dmm/ai/spark/dmm/km/RCS/fp.scala,v 1.1 2015/05/05 19:34:08 dmm Exp $
//

import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering._
import scala.collection.mutable.ArrayBuffer
import scala.math.sqrt
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.fpm.{FPGrowth, FPGrowthModel}

// the input data set containing all transactions

val kddcupData = "kddcup.data_10_percent"
val datafile   = "%s/%s".format(System.getProperty("user.dir"), kddcupData)
val rawData    = if (new java.io.File(datafile).exists) {
                     sc.textFile(datafile)
                 } else {
                     println("Datafile %s not found.\n".format(datafile))
                     sys.exit(1)
                 }

//
//	build transactions
//
//	buffer(1)  protocol_type: symbolic
//	buffer(2)  service: symbolic.
//	buffer(3)  flag: symbolic
//	buffer(6)  land: symbolic.
//	buffer(11) logged_in: discrete
//	buffer(20) is_host_login: discrete
//      buffer(21) is_guest_login: discrete
//
//	buffer(length-1) label 
//
//	Only 1-3 and label appear to be useful for FP Growth
//
//	Note: Items in a transaction must have unique values
//


val transactions = rawData.map {line =>
  val buffer = ArrayBuffer[String]()
  buffer.appendAll(line.split(","))
  Array(buffer(1),
        buffer(2),
	buffer(3),
        buffer(buffer.length-1))

}.cache()


//
//	build the model
//

val fpg = new FPGrowth()
fpg.setMinSupport(0.05)
fpg.setNumPartitions(10)
val model = fpg.run(transactions)

//
//	see what we've learned
//

model.freqItemsets.collect().foreach { itemset =>
  println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
}


