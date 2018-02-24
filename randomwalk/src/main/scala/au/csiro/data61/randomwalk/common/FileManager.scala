package au.csiro.data61.randomwalk.common

import java.io.{BufferedWriter, File, FileWriter}

import better.files._
import scala.io.Source
import scala.util.Try

/**
  * Created by Hooman on 2018-02-16.
  */
case class FileManager(config: Params) {

  def readFromFile(): Array[(Int, Array[(Int, Float)])] = {
    val lines = Source.fromFile(config.input).getLines.toArray

    lines.flatMap { triplet =>
      val parts = triplet.split("\\s+")
      // if the weights are not specified it sets it to 1.0

      val weight = config.weighted && parts.length > 2 match {
        case true => Try(parts.last.toFloat).getOrElse(1.0f)
        case false => 1.0f
      }

      val (src, dst) = (parts.head.toInt, parts(1).toInt)
      if (config.directed) {
        Array((src, Array((dst, weight))), (dst, Array.empty[(Int, Float)]))
      } else {
        Array((src, Array((dst, weight))), (dst, Array((src, weight))))
      }
    }.groupBy(_._1).map { case (src, edges) =>
      var neighbors = Array.empty[(Int, Float)]
      edges.foreach(neighbors ++= _._2)
      (src, neighbors)
    }.toArray
  }

  def save(probs: Array[Array[Double]]): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${Property.probSuffix}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(probs.map(array => array.map(a => f"$a%1.4f").mkString("\t")).mkString("\n"))
    bw.flush()
    bw.close()
  }

  def save(vertices: Array[Int], numSteps: Array[Array[Int]], suffix: String): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.rrType}-$suffix-wl${
      config.walkLength}-nw${config.numWalks}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(s"${vertices.mkString("\t")}\n")
    for (steps <- numSteps) {
      bw.write(s"${steps.mkString("\t")}\n")
    }
    bw.flush()
    bw.close()

  }

  def save(paths: Array[Array[Int]], suffix: String): Array[Array[Int]] = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.cmd}-$suffix.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(paths.map { case (path) =>
        val pathString = path.mkString("\t")
        s"$pathString"
    }.mkString("\n"))
    bw.flush()
    bw.close()
    paths
  }

  def save(paths: Array[Array[Int]]): Array[Array[Int]] = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${Property.pathSuffix}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(paths.map {
      case (path) =>
        val pathString = path.mkString("\t")
        s"$pathString"
    }.mkString("\n"))
    bw.flush()
    bw.close()
    paths
  }

  def saveCounts(counts: Array[(Int, (Int, Int))]) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${Property.countsSuffix}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(counts.sortWith(_._2._2 > _._2._2).map {
      case (vId, (count, occurs)) =>
        s"$vId\t$count\t$occurs"
    }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def save(degrees: Array[(Int, Int)]) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${Property.degreeSuffix}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(degrees.map { case (v, d) => s"$v\t$d" }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveAffecteds(afs: Array[(Int, Array[Int])]) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${Property.affecteds}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(afs.map { case (vId, af) => s"$vId\t${af.mkString("\t")}" }.mkString("\n"))
    bw.flush()
    bw.close()
  }
}
