package au.csiro.data61.randomwalk.common

import au.csiro.data61.randomwalk.algorithm.{GraphMap, RandomSample}
import com.sun.org.apache.bcel.internal.generic.Type

import scala.collection.mutable

/**
  * Created by Hooman on 2018-03-06.
  */
object GraphUtils {

  def computeSecondOrderProbs(config: Params): (mutable.HashMap[(Int, Int), Int], Seq[(Int, Int,
    Double)]) = {
    val vertices = GraphMap.getVertices()
    val edgeIds = new mutable.HashMap[(Int, Int), Int]
    var id = 0
    for (v <- vertices) {
      val neighbors = GraphMap.getNeighbors(v)
      for (e <- neighbors) {
        val dst = e._1
        edgeIds.put((v, dst), id)
        id += 1
      }
    }

    val edges = edgeIds.keySet
    val rSample = RandomSample()
    val soProbs = edges.map { case (prev, curr) =>
      val srcEdge:Int = edgeIds.getOrElse((prev, curr), throw new Exception(s"Edge $prev -> $curr is not found.") )
      val prevNeighbors = GraphMap.getNeighbors(prev)
      val currNeighbors = GraphMap.getNeighbors(curr)
      val unNormProbs = rSample.computeSecondOrderWeights(p = config.p, q = config.q, prev,
        prevNeighbors, currNeighbors)
      val sum = unNormProbs.foldLeft(0.0) { case (w1, (_, w2)) => w1 + w2 }
      var probs = Seq.empty[(Int, Int, Double)]
      unNormProbs.foreach { case (dst, w) =>
        val dstEdge:Int = edgeIds.getOrElse((curr, dst), throw new Exception(s"Edge $curr -> $dst is not found."))
        probs ++= Seq((srcEdge, dstEdge, w / sum))
      }
      probs
    }.flatten.toSeq

    (edgeIds, soProbs)

  }

}
