package au.csiro.data61.randomwalk.algorithm

import java.util

import au.csiro.data61.randomwalk.common.{FileManager, Params}
import org.apache.log4j.LogManager

import scala.collection.mutable
import scala.collection.parallel.ParSeq
import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

case class UniformRandomWalk(config: Params) extends Serializable {

  def computeAffecteds(vertices: Seq[Int], affectedLength: Int): Seq[(Int, Array[Int])] = {

    vertices.map { v =>
      def computeAffecteds(afs: Array[Int], visited: util.HashSet[Int], v: Int, al: Int,
                           length: Int): Unit = {
        if (length >= al)
          return
        visited.add(v)
        val neighbors = GraphMap.getNeighbors(v)
        //        if (neighbors != null) {
        for (n <- neighbors) {
          if (!visited.contains(n._1)) {
            afs(length) += 1
            visited.add(n._1)
            computeAffecteds(afs, visited, n._1, al, length + 1)
          }
        }
        //        }
      }

      val affecteds = new Array[Int](affectedLength)
      val visited = new util.HashSet[Int]()
      visited.add(v)
      affecteds(0) = 1
      computeAffecteds(affecteds, visited, v, affectedLength, 0)

      (v, affecteds)
    }.sortBy(_._2.last)
  }

  def computeProbs(paths: Seq[Seq[Int]]): Array[Array[Double]] = {
    val n = GraphMap.getVertices().length
    val matrix = Array.ofDim[Double](n, n)
    paths.foreach { case p =>
      for (i <- 0 until p.length - 1) {
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


  lazy val logger = LogManager.getLogger("rwLogger")
  var nVertices: Int = 0
  var nEdges: Int = 0

  def execute(): ParSeq[(Int, Int, Seq[Int])] = {
    firstOrderWalk(loadGraph())
  }

  /**
    * Loads the graph and computes the probabilities to go from each vertex to its neighbors
    *
    * @return
    */
  def loadGraph(): ParSeq[(Int, (Int, Int, Seq[Int]))] = {

    val g: ParSeq[(Int, mutable.Set[(Int, Float)])] = FileManager(config).readFromFile(config
      .directed)
    initRandomWalk(g)
  }

//  def checkGraphMap() = {
//    //    save(degrees())
//    println(degrees().sortBy(_._1).map { case (v, d) => s"$v\t$d" }.mkString("\n"))
//    for (v <- GraphMap.getVertices().sortBy(a => a)) {
//      val n = GraphMap.getNeighbors(v).map(_._1)
//      println(s"$v -> ${n.mkString(" ")}")
//    }
//  }

  def initWalker(v: Int): Seq[(Int, (Int, Int, Seq[Int]))] = {
    Seq.fill(config.numWalks)(Seq((v, (1, 1, Seq(v))))).flatten
  }


  def initRandomWalk(g: ParSeq[(Int, mutable.Set[(Int, Float)])]): ParSeq[(Int, (Int, Int, Seq[Int]))]
  = {
    println("****** Initializing random walk ******")
    buildGraphMap(g.seq)

    nVertices = g.length
    nEdges = 0
    nEdges = g.foldLeft(0)(_ + _._2.size)

    //    logger.info(s"edges: $nEdges")
    //    logger.info(s"vertices: $nVertices")
    println(s"edges: $nEdges")
    println(s"vertices: $nVertices")

    createWalkers(g)
  }

  def createWalkers(g: ParSeq[(Int, mutable.Set[(Int, Float)])]): ParSeq[(Int, (Int, Int, Seq[Int]))] = {
    g.flatMap {
      case (vId: Int, _) => Seq.fill(config.numWalks)((vId, (1, 1, Seq(vId))))
    }
  }

  def createWalkersByVertices(vertices: ParSeq[Int]): ParSeq[(Int, (Int, Int, Seq[Int]))] = {
    vertices.flatMap { case (vId) => Seq.fill(config.numWalks)((vId, (1, 1, Seq(vId)))) }
  }

  def firstOrderWalk(initPaths: ParSeq[(Int, (Int, Int, Seq[Int]))], nextFloat: () => Float = Random
    .nextFloat): ParSeq[(Int, Int, Seq[Int])] = {
    val walkLength = config.walkLength

    val paths: ParSeq[(Int, Int, Seq[Int])] = initPaths.map { case (_, steps) =>
      var path = steps._3
      val wVersion = steps._1
      val firstIndex = steps._2
      val rSample = RandomSample(nextFloat)
      breakable {
        while (path.length < walkLength) {
          val neighbors = GraphMap.getNeighbors(path.last)
          if (neighbors.size > 0) {
            val (nextStep, _) = rSample.sample(neighbors)
            path = path ++ Seq(nextStep)
          } else {
            break
          }
        }
      }
      (wVersion, firstIndex, path)
    }

    paths
  }

  def secondOrderWalk(initPaths: ParSeq[(Int, (Int, Int, Seq[Int]))], nextFloat: () => Float = Random
    .nextFloat): ParSeq[(Int, Int, Seq[Int])] = {
    println("%%%%% Starting random walk %%%%%")
    val walkLength = config.walkLength
    val paths: ParSeq[(Int, Int, Seq[Int])] = initPaths.map { case (_, s1) =>
      var init = s1._3
      val wVersion = s1._1
      val firstIndex = s1._2
      if (init.length == 1) {
        val rSample = RandomSample(nextFloat)
        val neighbors = GraphMap.getNeighbors(s1._3.head)
        if (neighbors.size > 0) {
          val (nextStep, _) = rSample.sample(neighbors)
          init = s1._3 ++ Seq(nextStep)
        }
      }
      (wVersion, firstIndex, init)
    }

    val walks = paths.map { case steps =>
      var path = steps._3
      val firstIndex = steps._2
      val wVersion = steps._1
      if (path.length > 1) {
        val rSample = RandomSample(nextFloat)
        breakable {
          while (path.length < walkLength) {
            val curr = path.last
            val prev = path(path.length - 2)
            val currNeighbors = GraphMap.getNeighbors(curr)
            val prevNeighbors = GraphMap.getNeighbors(prev)
            if (currNeighbors.size > 0) {
              val (nextStep, _) = rSample.secondOrderSample(p = config.p, q = config.q, prevId =
                prev, prevNeighbors = prevNeighbors, currNeighbors = currNeighbors)
              path = path ++ Seq(nextStep)
            } else {
              break
            }
          }
        }
      }
      (wVersion, firstIndex, path)
    }
    println("%%%%% Finished random walk %%%%%")
    walks
  }

  def secondOrderWalkWitIds(initPaths: ParSeq[(Int, (Int, Int, Seq[Int]))], nextFloat: () => Float =
  Random
    .nextFloat): ParSeq[(Int, (Int, Int, Seq[Int]))] = {
    println("%%%%% Starting random walk %%%%%")
    val walkLength = config.walkLength
    val paths: ParSeq[(Int, (Int, Int, Seq[Int]))] = initPaths.map { case s1 =>
      val id = s1._1
      val wVersion = s1._2._1
      val firstIndex = s1._2._2
      var init = s1._2._3
      if (init.length == 1) {
        val rSample = RandomSample(nextFloat)
        val neighbors = GraphMap.getNeighbors(s1._2._3.head)
        if (neighbors.size > 0) {
          val (nextStep, _) = rSample.sample(neighbors)
          init = s1._2._3 ++ Seq(nextStep)
        }
      }
      (id, (wVersion, firstIndex, init))
    }

    val walks = paths.map { case steps =>
      val id = steps._1
      val wVersion = steps._2._1
      val firstIndex = steps._2._2
      var path = steps._2._3
      if (path.length > 1) {
        val rSample = RandomSample(nextFloat)
        breakable {
          while (path.length < walkLength) {
            val curr = path.last
            val prev = path(path.length - 2)
            val currNeighbors = GraphMap.getNeighbors(curr)
            val prevNeighbors = GraphMap.getNeighbors(prev)
            if (currNeighbors.size > 0) {
              val (nextStep, _) = rSample.secondOrderSample(p = config.p, q = config.q, prevId =
                prev, prevNeighbors = prevNeighbors, currNeighbors = currNeighbors)
              path = path ++ Seq(nextStep)
            } else {
              break
            }
          }
        }
      }
      (id, (wVersion, firstIndex, path))
    }
    println("%%%%% Finished random walk %%%%%")
    walks
  }

  def buildGraphMap(graph: Seq[(Int, mutable.Set[(Int, Float)])]): Unit = {
    GraphMap.reset // This is only to run on a single executor.
    graph.foreach { case (vId, neighbors) =>
      GraphMap.addVertex(vId, neighbors)
    }

  }

  def queryPaths(paths: Seq[(Int, Int, Seq[Int])]): Seq[(Int, (Int, Int))] = {
    var nodes: Seq[Int] = Seq.empty[Int]
    var numOccurrences: Array[(Int, (Int, Int))] = null
    if (config.nodes.isEmpty) {
      nodes = GraphMap.getVertices()
    } else {
      nodes = config.nodes.split("\\s+").map(s => s.toInt)
    }

    numOccurrences = new Array[(Int, (Int, Int))](nodes.length)

    for (i <- 0 until nodes.length) {
      numOccurrences(i) = (nodes(i),
        paths.map { case (_, _, steps) =>
          val counts = steps.count(s => s == nodes(i))
          val occurs = if (counts > 0) 1 else 0
          (counts, occurs)
        }.reduce((c, o) => (c._1 + o._1, c._2 + o._2)))
    }

    numOccurrences
  }

}
