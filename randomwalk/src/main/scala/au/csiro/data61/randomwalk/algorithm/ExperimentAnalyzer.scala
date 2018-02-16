package au.csiro.data61.randomwalk.algorithm

import java.io.{BufferedWriter, FileWriter}
import java.util

import au.csiro.data61.randomwalk.common.Params
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable

/**
  * Created by Hooman on 2018-02-16.
  */
case class ExperimentAnalyzer(context: SparkContext, config: Params) extends Serializable {

  def readFile(f: String, nRun: Int): Array[RDD[Array[Int]]] = {
    val paths = new Array[RDD[Array[Int]]](nRun)

    for (i <- 0 until nRun) {
      val fName = f + s"-$i"
      println(s"reading $fName")
      paths(i) = context.textFile(fName, minPartitions
        = config
        .rddPartitions).map { triplet =>
        triplet.split("\\s+").map(p => p.toInt)
      }
      println(s"Number of paths: ${paths(i).count()}")
    }
    paths
  }

  def save(all_means: util.LinkedList[Double], m: String) = {
    val output = "/Users/Ganymedian/Desktop/dynamic-rw/" + m + "-errors.txt"
    val bw = new BufferedWriter(new FileWriter(output))
    for (i <-0 until all_means.size()) {
      bw.write(s"${all_means.get(i)}\n")
    }

    bw.flush()
    bw.close()
  }

  def analyze(degrees: Array[(Int, Int)]): Unit = {
    val karate = "/Users/Ganymedian/Desktop/Projects/stellar-random-walk-research/randomwalk/src/test/resources/karate.txt"
    val pFile = "/Users/Ganymedian/Desktop/dynamic-rw/karate-100-100/ar"
    val methods = Array("m1", "m2", "m3", "m4")
    val v = "v30"
    val bucket = 20;

    val tps = Array.ofDim[Double](34, 34)
    for (d <- degrees) {
      val neighbors = GraphMap.getNeighbors(d._1)
      val tp = 1.0 / d._2
      for (dst <- neighbors) {
        tps(d._1 - 1)(dst._1 - 1) = tp
      }
    }
    for (m <- methods) {
      var fName = pFile
      fName = m match {
        case "m1" => fName + "/m1"
        case _ => fName + s"/$m-$v"
      }
      val paths = readFile(fName, config.numRuns)
      val all_means = new util.LinkedList[Double]()
      val pLength = new util.LinkedList[Int]()
      for (bb <- 0 until bucket) {
        val l = 5 * (bb + 1)
        val all_errors = new util.LinkedList[Double]()
        for (ps <- paths) {
          val probs = computeProbs(ps.collect(), l)
          val errors = computErrors(tps, probs)
          all_errors.add(errors.flatten.max)
        }
        var sum = 0.0
        for (e <- 0 until all_errors.size()) {
          sum += all_errors.get(e)
        }
        all_means.add(sum / all_errors.size())
      }
      save(all_means, m)
    }
  }

  def computErrors(tps1: Array[Array[Double]], tps2: Array[Array[Double]]) = {
    val errors = Array.ofDim[Double](34, 34)
    for (i <- 0 until tps1.length)
      for (j <- 0 until tps2.length) {
        errors(i)(j) = Math.abs(tps1(i)(j) - tps2(i)(j))
      }
    errors
  }

  def computeProbs(paths: Array[Array[Int]], l: Int): Array[Array[Double]] = {
    val n = 34
    val matrix = Array.ofDim[Double](n, n)
    paths.foreach { case p =>
      for (i <- 0 until l-1) {
        matrix(p(i) - 1)(p(i + 1) - 1) += 1
      }
    }

    matrix.map { row =>
      val sum = row.sum
      row.map { o =>
        o / sum.toDouble
      }
    }
  }

}
