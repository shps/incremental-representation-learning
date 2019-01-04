package au.csiro.data61.randomwalk.algorithm

import scala.collection.mutable
import scala.util.Random

case class RandomSample(nextFloat: () => Float = Random.nextFloat) {


  /**
    *
    * @return
    */
  final def sample(edges: mutable.Set[(Int, Float)]): (Int, Float) = {

    val sum = edges.foldLeft(0.0) { case (w1, (_, w2)) => w1 + w2 }

    val p = nextFloat()
    var acc = 0.0
    for ((dstId, w) <- edges) {
      acc += w / sum
      if (acc >= p)
        return (dstId, w)
    }

    edges.head
  }

  final def computeSecondOrderWeights(p: Float = 1.0f,
                                      q: Float = 1.0f,
                                      prevId: Int,
                                      prevNeighbors: mutable.Set[(Int, Float)],
                                      currNeighbors: mutable.Set[(Int, Float)]): mutable.Set[
    (Int, Float)
    ] = {
    currNeighbors.map { case (dstId, w) =>
      var unnormProb = w / q
      if (dstId == prevId) unnormProb = w / p
      else {
        if (prevNeighbors.exists(_._1 == dstId)) unnormProb = w
      }
      (dstId, unnormProb)
    }
  }

  /**
    *
    * @param p
    * @param q
    * @param prevId
    * @param prevNeighbors
    * @param currNeighbors
    * @return
    */
  final def secondOrderSample(p: Float = 1.0f,
                              q: Float = 1.0f,
                              prevId: Int,
                              prevNeighbors: mutable.Set[(Int, Float)],
                              currNeighbors: mutable.Set[(Int, Float)]): (Int, Float) = {
    val newCurrentNeighbors = computeSecondOrderWeights(p, q, prevId, prevNeighbors, currNeighbors)
    sample(newCurrentNeighbors)
  }
}
