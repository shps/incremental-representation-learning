package au.csiro.data61.randomwalk.common

import java.io.{BufferedWriter, File, FileWriter}

import org.apache.spark.{HashPartitioner, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.util.Try

/**
  * Created by Hooman on 2018-02-16.
  */
case class FileManager(context: SparkContext, config: Params) {

  lazy val partitioner: HashPartitioner = new HashPartitioner(config.rddPartitions)

  def readFromFile(): RDD[(Int, Array[(Int, Float)])] = {
    // the directed and weighted parameters are only used for building the graph object.
    // is directed? they will be shared among stages and executors
    val bcDirected = context.broadcast(config.directed)
    val bcWeighted = context.broadcast(config.weighted) // is weighted?
    context.textFile(config.input, minPartitions
      = config
      .rddPartitions).flatMap { triplet =>
      val parts = triplet.split("\\s+")
      // if the weights are not specified it sets it to 1.0

      val weight = bcWeighted.value && parts.length > 2 match {
        case true => Try(parts.last.toFloat).getOrElse(1.0f)
        case false => 1.0f
      }

      val (src, dst) = (parts.head.toInt, parts(1).toInt)
      if (bcDirected.value) {
        Array((src, Array((dst, weight))), (dst, Array.empty[(Int, Float)]))
      } else {
        Array((src, Array((dst, weight))), (dst, Array((src, weight))))
      }
    }.
      reduceByKey(_ ++ _).
      partitionBy(partitioner).
      persist(StorageLevel.MEMORY_AND_DISK)
  }

  def save(probs: Array[Array[Double]]): Unit = {
    val file = new File(s"${config.output}.${Property.probSuffix}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(probs.map(array => array.map(a => f"$a%1.4f").mkString("\t")).mkString("\n"))
    bw.flush()
    bw.close()
  }

  def save(vertices: Array[Int], numSteps: Array[Array[Int]], suffix:String): Unit = {
    val file = new File(s"${config.output}/${config.rrType}-$suffix-wl${config.walkLength}-nw${config.numWalks}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(s"${vertices.mkString("\t")}\n")
    for (steps <- numSteps) {
      bw.write(s"${steps.mkString("\t")}\n")
    }
    bw.flush()
    bw.close()

  }

  def save(paths: RDD[Array[Int]], suffix: String): RDD[Array[Int]] = {

    paths.map {
      case (path) =>
        val pathString = path.mkString("\t")
        s"$pathString"
    }.repartition(1).saveAsTextFile(s"${config.output}/${config.cmd}/$suffix")
    paths
  }

  def save(paths: RDD[Array[Int]]): RDD[Array[Int]] = {

    paths.map {
      case (path) =>
        val pathString = path.mkString("\t")
        s"$pathString"
    }.repartition(config.rddPartitions).saveAsTextFile(s"${config.output}.${Property.pathSuffix}")
    paths
  }

  def saveCounts(counts: Array[(Int, (Int, Int))]) = {

    context.parallelize(counts, config.rddPartitions).sortBy(_._2._2, ascending = false).map {
      case (vId, (count, occurs)) =>
        s"$vId\t$count\t$occurs"
    }.repartition(1).saveAsTextFile(s"${config.output}.${Property.countsSuffix}")
  }

  def save(degrees: Array[(Int, Int)]) = {
    val file = new File(s"${config.output}/${Property.degreeSuffix}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(degrees.map { case (v, d) => s"$v\t$d" }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveAffecteds(afs: RDD[(Int, Array[Int])]) = {
    afs.map {
      case (vId, af) =>
        s"$vId\t${af.mkString("\t")}"
    }.repartition(1).saveAsTextFile(s"${config.output}.${Property.affecteds}")
  }
}
