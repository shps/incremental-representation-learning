package au.csiro.data61.randomwalk.common

import java.io.{BufferedWriter, File, FileWriter}

import better.files._

import scala.collection.mutable
import scala.collection.parallel.{ParSeq, immutable}
import scala.io.Source
import scala.util.{Random, Try}


/**
  * Created by Hooman on 2018-02-16.
  */
case class FileManager(config: Params) {

  def readCountFile(): ParSeq[(Int, Int)] = {
    val lines = Source.fromFile(config.input).getLines.toArray.par

    lines.flatMap { triplet =>
      val parts = triplet.split(config.delimiter)

      Seq((parts.head.toInt, parts(1).toInt))
    }
  }

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

  def readJsonFile(): ParSeq[(String, String, Int)] = {
    val lines = Source.fromFile(config.input).getLines.toList.par
    val toRemove = "[\"]".toSet
    val toRemove2 = ",".toSet
    lines.flatMap { case l =>
      val fl = l.filterNot(toRemove).split(", ")
      if (fl.length > 1) {
        val date = fl(2).filterNot(toRemove2) match {
          case "null" => 0
          case other => other.toInt
        }
        Seq((fl(0), fl(1), date))
      } else {
        Seq.empty[(String, String, Int)]
      }
    }
  }

  def readEdgeList(): ParSeq[(Int, Int)] = {
    val lines = Source.fromFile(config.input).getLines.toArray.par

    lines.flatMap { triplet =>
      val parts = triplet.split(config.delimiter)

      Seq((parts.head.toInt, parts(1).toInt))
    }.distinct

  }

  def convertDelimiter() = {
    println(s"Convert from del ${config.delimiter} to del ${config.delimiter2}")
//    var d2 = config.delimiter2
//    if (d2.contains("\\s+"))
//      d2 = "\t"
    val lines = Source.fromFile(config.input).getLines.toArray.par.map { triplet =>
      triplet.split(config.delimiter).mkString("\t")
    }

    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/converted.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(lines.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def readPartitionedEdgeList(): Seq[(Int, Seq[(Int, Int)])] = {
    val lines = readEdgeList().seq
    val edgePerPartition: Int = Math.max(lines.size / 10000, 1)
    println(s"Number of edges per step: $edgePerPartition")
    Random.setSeed(config.seed)

    Random.shuffle(lines).sliding(edgePerPartition, edgePerPartition).toSeq
      .zipWithIndex.map(a => (a._2, a._1))
  }

  def readPartitionEdgeListWithInitEdges(seedIncrement: Int): (Seq[(Int, Int)], Seq[(Int, Seq[
    (Int, Int)])]) = {
    val lines = readEdgeList().seq
    val edgePerPartition: Int = Math.max((lines.size * config.edgeStreamSize).toInt, 1)
    println(s"Number of edges per step: $edgePerPartition")
    Random.setSeed(config.seed + seedIncrement)

    val (part1, part2) = Random.shuffle(lines).splitAt((lines.size * config.initEdgeSize).toInt)
    (part1, part2.sliding(edgePerPartition, edgePerPartition).toSeq
      .zipWithIndex.map(a => (a._2 + 1, a._1)))
  }

  def readEdgeListByYear(): Seq[(Int, Seq[(Int, Int)])] = {
    val lines = Source.fromFile(config.input).getLines.toSeq.par
    lines.flatMap { triplet =>
      val parts = triplet.split(config.delimiter)

      Seq((parts.head.toInt, parts(1).toInt, parts(2).toInt))
    }.groupBy(_._3).toSeq.map { case (year, tuple) =>
      (year, tuple.map(a => (a._1, a._2)).seq)
    }.seq.sortWith(_._1 < _._1)
  }

  def readAlreadyPartitionedEdgeList(): (Seq[(Int, Int)], Seq[(Int, Seq[
    (Int, Int)])]) = {
    val lines = Source.fromFile(config.input).getLines.toSeq.par
    val partitions = lines.flatMap { triplet =>
      val parts = triplet.split(config.delimiter)

      Seq((parts.head.toInt, parts(1).toInt, parts(2).toInt))
    }.groupBy(_._3).toSeq.map { case (year, tuple) =>
      (year, tuple.map(a => (a._1, a._2)).seq)
    }.seq.sortWith(_._1 < _._1)
    val (part1, part2) = partitions.splitAt((partitions.size * config.initEdgeSize).toInt)
    (part1.flatMap(_._2), part2)
  }

  def saveProbs(probs: Seq[Seq[Double]]): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${Property.probSuffix}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(probs.map(array => array.map(a => f"$a%1.4f").mkString("\t")).mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveNumSteps(vertices: Seq[Int], numSteps: Array[Array[Int]], suffix: String): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.rrType}-$suffix-wl${
      config.walkLength
    }-nw${config.numWalks}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(s"${vertices.mkString("\t")}\n")
    for (steps <- numSteps) {
      bw.write(s"${steps.mkString("\t")}\n")
    }
    bw.flush()
    bw.close()

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
    bw.write(pairs.map { case (t, c) => s"$t\t$c" }.mkString("\n"))
    bw.flush()
    bw.close()

    val file2 = new File(s"${config.output}/${config.cmd}-vocabs-$suffix.txt")
    val bw2 = new BufferedWriter(new FileWriter(file2))
    bw2.write(vocabs.mkString("\n"))
    bw2.flush()
    bw2.close()
  }

  def savePaths(paths: ParSeq[(Int, Int, Seq[Int])], suffix: String): ParSeq[(Int, Int, Seq[Int])
    ] = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.cmd}-$suffix.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(paths.map { case (wVersion, firstIndex, path) =>
      val pathString = s"$wVersion\t$firstIndex\t" + path.mkString("\t")
      s"$pathString"
    }.mkString("\n"))
    bw.flush()
    bw.close()
    paths
  }

  def saveEdgeList(edges: Seq[(Int, (Int, Float))], suffix: String) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.rrType}-$suffix.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(edges.map { case (edge) =>
      s"${edge._1}\t${edge._2._1}"
    }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def savePaths(paths: ParSeq[(Int, Int, Seq[Int])]): ParSeq[(Int, Int, Seq[Int])] = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${Property.pathSuffix}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(paths.map {
      case (wVersion, firstIndex, path) =>
        val pathString = s"$wVersion\t$firstIndex\t" + path.mkString("\t")
        s"$pathString"
    }.mkString("\n"))
    bw.flush()
    bw.close()
    paths
  }

  def saveAffectedVertices(afs: mutable.HashSet[Int], suffix: String) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${config.cmd}-$suffix.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(afs.toSeq.sortWith(_ < _).mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveCounts(counts: Seq[(Int, Int)], fName: String) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/$fName.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(counts.map {
      case (vId, count) =>
        s"$vId\t$count"
    }.mkString("\n"))
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

  def saveSecondOrderProbs(edgeIds: mutable.HashMap[(Int, Int), Int], probs: Seq[(Int, Int,
    Double)]) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${Property.edgeIds}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(edgeIds.map { case ((src, dst), id) => s"$src\t$dst\t$id" }.mkString("\n"))
    bw.flush()
    bw.close()

    val file2 = new File(s"${config.output}/${Property.soProbs}.txt")
    val bw2 = new BufferedWriter(new FileWriter(file2))
    bw2.write(probs.map { case (sId, dId, p) => s"$sId\t$dId\t$p" }.mkString("\n"))
    bw2.flush()
    bw2.close()
  }

  def saveAffecteds(afs: Seq[(Int, Array[Int])]) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/${Property.affecteds}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(afs.map { case (vId, af) => s"$vId\t${af.mkString("\t")}" }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveCoAuthors(tuples: List[(String, String, Int)]) = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/coauthors.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(tuples.par.map { case (a1, a2, year) => s"$a1,$a2,$year" }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveIds(authors: ParSeq[(String, Int)]): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/authors-ids.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(authors.map { case (a, id) => s"$a,$id" }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveCoAuthors(coauthors: ParSeq[(Int, Int, Int)]): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/coauthors-edge-list.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(coauthors.par.map { case (a1, a2, year) => s"$a1\t$a2\t$year" }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveNumAuthors(ua: String, fname: String): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/$fname.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(ua)
    bw.flush()
    bw.close()
  }

  def saveCountGroups(groups: Array[Int]): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/count-groups.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(groups.zipWithIndex.map { case (c, g) => s"$g\t$c" }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveLabels(labels: ParSeq[(Int, ParSeq[Int])]): Unit = {
    config.output.toFile.createIfNotExists(true)
    val file = new File(s"${config.output}/sorted-labels.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(labels.map { case (v, labels) => s"$v\t${labels.mkString("\t")}" }.mkString("\n"))
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
