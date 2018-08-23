package au.csiro.data61.randomwalk.common

import java.io.{BufferedWriter, File, FileWriter}

import better.files._

import scala.collection.mutable
import scala.collection.parallel.ParSeq
import scala.io.Source
import scala.util.{Random, Try}


/**
  * Created by Hooman on 2018-02-16.
  */
case class FileManager(config: Params) {


  def readWalks(): ParSeq[Seq[Int]] = {
    val lines = Source.fromFile(config.input).getLines.toArray.par

    lines.map { triplet =>
      triplet.split(config.delimiter).map(a => a.toInt).toSeq
    }
  }

  def readFromFile(directed: Boolean): ParSeq[(Int, mutable.Set[(Int, Float)])] = {
    val lines = Source.fromFile(config.input).getLines.toArray.par

    lines.flatMap { triplet =>
      val parts = triplet.split(config.delimiter)
      // if the weights are not specified it sets it to 1.0

      val weight = config.weighted && parts.length > 2 match {
        case true => Try(parts.last.toFloat).getOrElse(1.0f)
        case false => 1.0f
      }

      val (src, dst) = (parts.head.toInt, parts(1).toInt)
      if (directed) {
        mutable.Set((src, mutable.Set((dst, weight))), (dst, mutable.Set.empty[(Int, Float)]))
      } else {
        Seq((src, mutable.Set((dst, weight))), (dst, mutable.Set((src, weight))))
      }
    }.groupBy(_._1).map { case (src, edges) =>
      val neighbors = edges.foldLeft(mutable.Set.empty[(Int, Float)])(_ ++ _._2)
      (src, neighbors)
    }.toSeq
  }

  def readEdgeList(): ParSeq[(Int, Int)] = {
    val lines = Source.fromFile(config.input).getLines.toArray.par

    lines.flatMap { triplet =>
      val parts = triplet.split(config.delimiter)

      val src = parts.head.toInt
      val dst = parts(1).toInt
      if (src == dst) // remove self edges
        Seq.empty[(Int, Int)]
      else
        Seq((src, dst))
    }.distinct

  }

  def readAlreadyPartitionedEdgeList(): (Seq[(Int, Int)], Seq[(Int, Seq[
    (Int, Int)])]) = {
    val lines = Source.fromFile(config.input).getLines.toSeq.par
    val partitions = lines.flatMap { triplet =>
      val parts = triplet.split(config.delimiter)
      val src = parts.head.toInt
      val dst = parts(1).toInt
      if (src == dst)
        Seq.empty[(Int, Int, Int)]
      else
        Seq((src, dst, parts(2).toInt))
    }.groupBy(_._3).toSeq.map { case (year, tuple) =>
      (year, tuple.map(a => (a._1, a._2)).seq)
    }.seq.sortWith(_._1 < _._1)
    val (part1, part2) = partitions.splitAt(config.initEdgeSize)
    (part1.flatMap(_._2), part2)
  }

  def readPartitionEdgeListWithInitEdges(): (Seq[(Int, Int)], Seq[(Int, Seq[
    (Int, Int)])]) = {
    val lines = readEdgeList().seq
    println(s"Number of edges per step: ${config.edgeStreamSize}")

    val (part1, part2) = Random.shuffle(lines).splitAt(config.initEdgeSize)
    (part1, part2.grouped(config.edgeStreamSize).toSeq.zipWithIndex.map(a => (a._2 + 1, a._1)))
  }

  def saveComputations(numSteps: Array[Array[Int]], suffix: String): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.rrType}-$suffix-wl${
      config.walkLength
    }-nw${config.numWalks}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    for (steps <- numSteps) {
      bw.write(s"${steps.mkString("\t")}\n")
    }
    bw.flush()
    bw.close()

  }

  def saveTimeSteps(times: Array[Array[Long]], suffix: String): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.rrType}-$suffix-wl${
      config.walkLength
    }-nw${config.numWalks}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    for (t <- times) {
      bw.write(s"${t.mkString("\t")}\n")
    }
    bw.flush()
    bw.close()

  }


  def saveGraphStats(graphStats: Seq[(Int, Int, Int, Int)], suffix: String): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/$suffix.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(graphStats.map(a => s"${a._1}\t${a._2}\t${a._3}\t${a._4}").mkString("\n"))
    bw.flush()
    bw.close()

  }

  def saveErrors(numSteps: Array[Array[Double]], suffix: String): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.rrType}-$suffix-wl${
      config.walkLength
    }-nw${config.numWalks}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    for (steps <- numSteps) {
      bw.write(s"${steps.mkString("\t")}\n")
    }
    bw.flush()
    bw.close()

  }

  def saveTargetContextPairs(pairs: ParSeq[(Int, Int)], vocabs: ParSeq[Int], suffix: String) {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.cmd}-$suffix.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    for (pair <- pairs) {
      bw.write(s"${pair._1}\t${pair._2}\n")
    }
    bw.flush()
    bw.close()

    val file2 = new File(s"${config.output}/${config.cmd}-vocabs-$suffix.txt")
    val bw2 = new BufferedWriter(new FileWriter(file2))
    bw2.write(vocabs.mkString("\n"))
    bw2.flush()
    bw2.close()
  }

  def savePaths(paths: ParSeq[(Int, Int, Int, Seq[Int])], suffix: String) {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.cmd}-$suffix.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(paths.map { case (wVersion, firstIndex, isNew, path) =>
      val pathString = s"$wVersion\t$firstIndex\t$isNew\t" + path.mkString("\t")
      s"$pathString"
    }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveAffectedVertices(afs: mutable.HashSet[Int], suffix: String) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.cmd}-$suffix.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(afs.toSeq.sortWith(_ < _).mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveDensities(densities: Seq[Double], suffix: String): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/$suffix.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(densities.zipWithIndex.map { d => s"${d._2}\t${d._1}" }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveBiggestScc(bScc: Seq[Int], suffix: String) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.cmd}-$suffix.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(bScc.sortWith(_ < _).mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveDegrees(degrees: Seq[(Int, Int)], suffix: String) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/$suffix.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(degrees.map { case (v, d) => s"$v\t$d" }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveGraphStreamStats(results: Seq[(Int, Int, Int, Int, Int)], fName: String) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/$fName")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(results.map { case (t, uV, uE, nV, nE) => s"$t\t$uV\t$uE\t$nV\t$nE" }.mkString("\n"))
    bw.flush()
    bw.close()
  }
}
