package au.csiro.data61.randomwalk.algorithm

import au.csiro.data61.randomwalk.common.{FileManager, Params}
import org.apache.log4j.LogManager

import scala.collection.mutable
import scala.collection.parallel.ParSeq
import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

case class UniformRandomWalk(config: Params) {

  lazy val logger = LogManager.getLogger("rwLogger")
  var nVertices: Int = 0
  var nEdges: Int = 0

  /**
    *
    *
    * @return
    */
  def loadGraph(): ParSeq[(Int, (Int, Int, Seq[Int]))] = {

    val g: ParSeq[(Int, mutable.Set[(Int, Float)])] = FileManager(config).readFromFile(config
      .directed)
    initRandomWalk(g)
  }


  def initRandomWalk(g: ParSeq[(Int, mutable.Set[(Int, Float)])]): ParSeq[(Int, (Int, Int,
    Seq[Int]))]
  = {
    println("****** Initializing random walk ******")
    buildGraphMap(g.seq)

    nVertices = g.length
    nEdges = 0
    nEdges = g.foldLeft(0)(_ + _._2.size)

    println(s"edges: $nEdges")
    println(s"vertices: $nVertices")

    createWalkers(g)
  }

  def createWalkers(g: ParSeq[(Int, mutable.Set[(Int, Float)])]): ParSeq[(Int, (Int, Int,
    Seq[Int]))] = {
    g.flatMap {
      case (vId: Int, _) => Seq.fill(config.numWalks)((vId, (1, 1, Seq(vId))))
    }
  }

  def createWalkersByVertices(vertices: ParSeq[Int]): ParSeq[(Int, (Int, Int, Seq[Int]))] = {
    vertices.flatMap { case (vId) => Seq.fill(config.numWalks)((vId, (1, 1, Seq(vId)))) }
  }

  def secondOrderWalk(initPaths: ParSeq[(Int, (Int, Int, Seq[Int]))], nextFloat: () => Float =
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
    GraphMap.reset
    graph.foreach { case (vId, neighbors) =>
      GraphMap.addVertex(vId, neighbors)
    }

  }


}
