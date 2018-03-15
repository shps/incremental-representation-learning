package au.csiro.data61.randomwalk.common

import au.csiro.data61.randomwalk.algorithm.{GraphMap, RandomSample, UniformRandomWalk}
import org.scalatest.{BeforeAndAfter, FunSuite}

import scala.collection.mutable
import scala.collection.parallel.{ParMap, ParSeq}

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
  test("testComputeSecondOrderProbsWithNoId") {
    var config = Params(input = karate, directed = false, p = 1.0f, q = 1.0f)
    val rw = UniformRandomWalk(config)
    rw.loadGraph()
    val probs: ParMap[(Int, Int, Int), Double] = GraphUtils.computeSecondOrderProbsWithNoId(config)

    checkCorrectnessWithNoId(probs, config)

    config = Params(input = karate, directed = false, p = 0.5f, q = 2.0f)
    val probs2 = GraphUtils.computeSecondOrderProbsWithNoId(config)
    checkCorrectnessWithNoId(probs2, config)
  }

  test("testComputeErrors") {
    val v1 = 1
    val v2 = 2
    val v3 = 3
    val w = 1f
    val v1N = Seq((v2, w))
    val v2N = Seq((v1, w), (v3, w))
    val v3N = Seq((v2, w))
    val p1 = Seq(v1, v2, v1, v2, v1)
//    val p12 = Seq(v1, v2, v1, v2, v3)
    val p2 = Seq(v2, v1, v2, v3, v2)
//    val p22 = Seq(v2, v3, v2, v3, v2)
    val p3 = Seq(v3, v2, v3, v2, v1)
    val walks = ParSeq(p1, p2, p3)
    GraphMap.addVertex(v1, v1N)
    GraphMap.addVertex(v2, v2N)
    GraphMap.addVertex(v3, v3N)
    var config = Params(directed = false, p = 1.0f, q = 1.0f)
    val probs: ParMap[(Int, Int, Int), Double] = GraphUtils.computeSecondOrderProbsWithNoId(config)

    for (e <- probs.toSeq) {
      val p = e._1
      println(s"${p._1}-${p._2}-${p._3}\t${e._2}")
    }

    println("**********")
    val mProbs = GraphUtils.computeEmpiricalTransitionProbabilities(walks, probs.keySet)
    for (e <- mProbs) {
      val p = e._1
      println(s"${p._1}-${p._2}-${p._3}\t${e._2}")
    }

    val errors = GraphUtils.computeErrors(walks, config)
    println(s"Errors: ${errors.mkString("\t")}")
    val (mean, max) = GraphUtils.computeErrorsMeanAndMax(walks, config)

    println(mean, max)
//    assert(mean = )
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

  def checkCorrectnessWithNoId(probs: ParMap[(Int, Int, Int), Double],
                               config: Params) = {
    assert(probs.seq.forall { case ((prev, curr, dst), p) =>
      val prevNeighbors = GraphMap.getNeighbors(prev)
      val currNeighbors = GraphMap.getNeighbors(curr)
      val unProbs = RandomSample().computeSecondOrderWeights(config.p, config.q, prev,
        prevNeighbors, currNeighbors)
      val sum = unProbs.foldLeft(0.0) { case (w1, (_, w2)) => w1 + w2 }
      val w = unProbs.filter(_._1 == dst).head._2
      p == (w / sum)

    })
  }

}
