package au.csiro.data61.randomwalk.algorithm

import scala.collection.mutable
import scala.collection.mutable.HashMap


/**
  *
  */

object GraphMap {


  private lazy val srcVertexMap: mutable.Map[Int, mutable.Set[(Int, Float)]] = new HashMap[Int,
    mutable.Set[(Int, Float)]]()

  def addVertex(vId: Int, neighbors: mutable.Set[(Int, Float)]): Unit = synchronized {
    srcVertexMap.get(vId) match {
      case None => {
        srcVertexMap.put(vId, neighbors)
      }
      case Some(value) => value
    }
  }

  def putVertex(vId: Int, neighbors: mutable.Set[(Int, Float)]): Unit = synchronized {
    srcVertexMap.put(vId, neighbors)
  }

  def getVertices(): Seq[Int] = {
    val vertices: Seq[Int] = srcVertexMap.keys.toSeq
    vertices
  }

  def getNumVertices: Int = {
    srcVertexMap.size
  }

  def getNumEdges: Int = {
    srcVertexMap.values.foldLeft(0)(_ + _.size)
  }

  /**
    * The reset is mainly for the unit test purpose.
    */
  def reset {
    srcVertexMap.clear()
  }

  def getNeighbors(vid: Int): mutable.Set[(Int, Float)] = {
    srcVertexMap.getOrElse(vid, mutable.Set.empty[(Int, Float)])
  }
}
