package au.csiro.data61.randomwalk.common

import au.csiro.data61.randomwalk.algorithm.{GraphMap, RandomSample, UniformRandomWalk}
import org.scalatest.{BeforeAndAfter, FunSuite}

import scala.collection.mutable

/**
  * Created by Hooman on 2018-03-06.
  */
class GraphUtilsTest extends FunSuite with BeforeAndAfter {

  private val karate = "./src/test/resources/karate.txt"
  after {
    GraphMap.reset
  }

  test("testComputeSecondOrderProbs") {
    var config = Params(input = karate, directed = false, p = 1.0f, q = 1.0f)
    val rw = UniformRandomWalk(config)
    rw.loadGraph()
    val (edgeIds, probs) = GraphUtils.computeSecondOrderProbs(config)
    var edges = new mutable.HashMap[Int, (Int, Int)]()
    edgeIds.foreach { case (e, id) =>
      edges.put(id, e)
    }

    checkCorrectness(edges, probs, config)

    config = Params(input = karate, directed = false, p = 0.5f, q = 2.0f)
    val (edgeIds2, probs2) = GraphUtils.computeSecondOrderProbs(config)
    edges = new mutable.HashMap[Int, (Int, Int)]()
    edgeIds2.foreach { case (e, id) =>
      edges.put(id, e)
    }

    checkCorrectness(edges, probs2, config)
  }

  def checkCorrectness(edges: mutable.HashMap[Int, (Int, Int)], probs: Seq[(Int, Int, Double)],
                       config: Params) = {
    assert(probs.forall { case (sEdgeId, dEdgeId, p) =>
      val (prev, curr) = edges.getOrElse(sEdgeId, null)
      val (curr2, dst) = edges.getOrElse(dEdgeId, null)
      val prevNeighbors = GraphMap.getNeighbors(prev)
      val currNeighbors = GraphMap.getNeighbors(curr)
      val unProbs = RandomSample().computeSecondOrderWeights(config.p, config.q, prev,
        prevNeighbors, currNeighbors)
      val sum = unProbs.foldLeft(0.0) { case (w1, (_, w2)) => w1 + w2 }
      val w = unProbs.filter(_._1 == dst).head._2
      p == (w / sum) && curr == curr2

    })
  }

}
