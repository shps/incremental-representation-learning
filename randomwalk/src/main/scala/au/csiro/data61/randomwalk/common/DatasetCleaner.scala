package au.csiro.data61.randomwalk.common

import java.util

import au.csiro.data61.randomwalk.algorithm.{GraphMap, UniformRandomWalk}

import scala.collection.mutable
import scala.collection.parallel.ParSeq


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
        comps ++= Seq(comp)
      }
    }

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

  def convertToUndirected(): ParSeq[(Int, Int)] = {
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
