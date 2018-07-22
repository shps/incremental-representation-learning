package au.csiro.data61.randomwalk.common

import java.util

import au.csiro.data61.randomwalk.algorithm.{GraphMap, UniformRandomWalk}

import scala.collection.mutable
import scala.collection.parallel.ParSeq
import scala.util.Random


/**
  * Created by Hooman on 2018-02-27.
  */
object DatasetCleaner {

  def checkRedundantEdges(edges: ParSeq[(Int, Int)], config: Params) = {
    val selfEdges = edges.count { case (a, b) => a == b }
    if (selfEdges > 0)
      throw new Exception(s"Number of self edges: $selfEdges")
    val rEdges = edges.flatMap { case (a, b) => Seq((a, b), (b, a)) }.distinct
    if (!config.directed)
      if (edges.size * 2 != rEdges.size) {
        throw new Exception(s"There are reversed edges. Expected: ${edges.size * 2}, Actual: " +
          s"${rEdges.size}")
      }
  }

  def checkDataSet(config: Params, initId: Int): Unit = {
    val fm = FileManager(config)
    val edges = fm.readEdgeList()
    UniformRandomWalk(config).loadGraph()
    println(s"Number of edges: ${edges.size}")
    val deduplicated = edges.distinct
    println(s"Number of edges after deduplication: ${deduplicated.size}")
    if (deduplicated.size != edges.size) {
      val duplicates = edges.groupBy(identity).filter(_._2.length > 1).map(_._1)
      println(s"Number of duplicates: ${duplicates.size}")
      println(s"${duplicates.map { case (s, d) => s"$s\t$d" }.mkString("\n")}")
    }
    checkRedundantEdges(edges, config)
    val vertices = edges.flatMap { case e => Seq(e._1, e._2) }.distinct.seq.sortWith(_ < _)
    println(s"Number of vertices: ${vertices.size}")
    for (i <- 0 until vertices.size) {
      val expected = initId + i
      if (vertices(i) != expected) {
        throw new Exception(s"Vertex ID ${vertices(i)} not equal the expected ID ${expected}")
      }
    }
  }

  def convertJsonFile(config: Params) {
    val fm = FileManager(config)
    val coAuthors = fm.readJsonFile()
    val filtered = coAuthors.filter(_._3 != 0).map { case (src, dst, y) =>
      if (src < dst)
        (src, dst, y)
      else
        (dst, src, y)
    }.groupBy { case (src, dst, _) => (src, dst)
    }.toSeq.map { case ((src, dst), duplicates) =>
      (src, dst, duplicates.seq.sortWith(_._3 < _._3).head._3)
    }.filter { case (a, b, _) => !a.contentEquals(b) }

    val authors = filtered.flatMap { case (src, dst, _) => Seq(src, dst) }
      .distinct.zipWithIndex
    fm.saveIds(authors)
    val idMaps = authors.toMap
    val coAuthorIded = filtered.map { case (a1, a2, y) =>
      val src = idMaps.getOrElse(a1, throw new Exception)
      val dst = idMaps.getOrElse(a2, throw new Exception)
      if (src < dst)
        (src, dst, y)
      else
        (dst, src, y)
    }.seq.sortWith(_._3 < _._3).par
    // saves as indirected, removes duplicates.
    fm.saveCoAuthors(coAuthorIded)
    val groups = coAuthorIded.groupBy(_._3).toSeq.seq.sortWith(_._1 < _._1).par

    GraphMap.reset
    var results = Seq.empty[(Int, Int, Int, Int, Int)]
    for (updates <- groups.seq) {
      val year = updates._1
      val edges = updates._2.distinct
      val uVertices = edges.flatMap { case (src, dst, _) => Seq(src, dst) }.distinct.length
      val nBefore = GraphMap.getNumVertices
      val mBefore = GraphMap.getNumEdges
      for (e <- edges) {
        GraphMap.addUndirectedEdge(e._1, e._2, 1f)
      }
      val nAfter = GraphMap.getNumVertices
      val mAfter = GraphMap.getNumEdges
      val newVertices = nAfter - nBefore
      val newEdges = mAfter - mBefore
      results ++= Seq((year, uVertices, edges.length, newVertices, newEdges / 2))
    }

    fm.saveGraphStreamStats(results.sortWith(_._1 < _._1), "streaming-stats.txt")

    val sums = groups.map { case (y, edges) =>
      (y, edges.length)
    }.seq.sortWith(_._1 < _._1).map { case (y, n) => s"$y\t$n" }.mkString("\n")
    fm.saveNumAuthors(sums, "coauthors-per-year")

    val ua = groups.map { case (y, edges) =>
      val count = edges.flatMap { case (src, dst, _) => Seq(src, dst) }.distinct.length
      (y, count)
    }.seq.sortWith(_._1 < _._1).map { case (y, n) => s"$y\t$n" }.mkString("\n")
    fm.saveNumAuthors(ua, "unique-authors-per-year")

  }

  def convertDelimiter(config: Params): Unit = {

  }

  def dfs(v: Int, vertices: mutable.Map[Int, Boolean]): Seq[Int] = {
    vertices(v) = true
    var compVertices = Seq(v)
    val neighbors = GraphMap.getNeighbors(v).toIterator
    while (neighbors.hasNext) {
      val next = neighbors.next()._1
      if (!vertices(next)) {
        compVertices ++= dfs(next, vertices)
      }
    }
    compVertices
  }

  def dfsNonRecurse(v: Int, vertices: mutable.Map[Int, Boolean]): Seq[Int] = {

    var compVertices = Seq.empty[Int]
    val stack: util.Stack[Int] = new util.Stack[Int]()
    stack.push(v)

    while (!stack.isEmpty) {
      val next = stack.pop()
      if (!vertices(next)) {
        vertices(next) = true
        compVertices ++= Seq(next)
        val neighbors = GraphMap.getNeighbors(next).toIterator
        while (neighbors.hasNext) {
          val n = neighbors.next()._1
          if (!vertices(n)) {
            stack.push(n)
          }
        }
      }
    }

    compVertices
  }

  def countNumberOfSCCs(): Seq[Int] = {
    val vertices = mutable.Map(GraphMap.getVertices().map(a => (a, false)): _*)
    var components = Seq.empty[Int]
    for (v <- vertices.keys) {
      if (!vertices(v)) {
        val comp = dfsNonRecurse(v, vertices)
        components ++= Seq(comp.size)
        println(s"Component ${components.size}\tsize: ${comp.size}")
      }
    }

    return components
  }

  def getBiggestSccAndCounts(): (Seq[Seq[Int]], Int) = {
    val vertices = mutable.Map(GraphMap.getVertices().map(a => (a, false)): _*)
    var components = 0
    var comps = Seq.empty[Seq[Int]]
    for (v <- vertices.keys) {
      if (!vertices(v)) {
        components += 1
        val comp = dfsNonRecurse(v, vertices)
//        println(s"Component $components\tsize: ${comp.size}")
        comps ++= Seq(comp)
      }
    }

//    var nVertices = 0
//    for (c <- comps) {
//      nVertices += c.size
//    }
//    assert(nVertices == GraphMap.getNumVertices)
    var max = 0
    var maxSize = comps(0).size
    for (i <- 1 until comps.length) {
      if (comps(i).size > maxSize) {
        max = i
        maxSize = comps(i).size
      }
    }

    return (comps, max)
  }

  //  def generateRandomConnectedEdgeStream(): Unit ={
  //
  //    val vertices = GraphMap.getVertices()
  //    Random.setSeed(1234)
  //    val v = Random.shuffle(vertices).head
  //
  //
  //
  //  }

  def convertToUndirected():ParSeq[(Int, Int)] = {
    val directedEdges = GraphMap.getVertices().par.flatMap { case v =>
      GraphMap.getNeighbors(v).flatMap { case (dst, _) =>
        if (v < dst)
          Seq((v, dst))
        else
          Seq((dst, v))
      }
    }

    println(s"Number of directed edges: ${directedEdges.size}")
    return directedEdges.distinct
  }

}
