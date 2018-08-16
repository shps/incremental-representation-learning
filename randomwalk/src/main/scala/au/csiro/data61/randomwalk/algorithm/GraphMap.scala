package au.csiro.data61.randomwalk.algorithm

import scala.collection.mutable
import scala.collection.mutable.HashMap


/**
  *
  */

object GraphMap {

  /**
    *
    * @param src
    * @param dst
    * @param w
    * @return true if the edge does not exist before. false if it updates an existing edge.
    */
  def addUndirectedEdge(src: Int, dst: Int, w: Float): Boolean = synchronized {
    val sNeighbors = GraphMap.getNeighbors(src)
    val dNeighbors = GraphMap.getNeighbors(dst)
    sNeighbors.add((dst, w))
    val isNew = dNeighbors.add((src, w))
    GraphMap.putVertex(src, sNeighbors)
    GraphMap.putVertex(dst, dNeighbors)
    isNew
  }


  private lazy val srcVertexMap: mutable.Map[Int, mutable.Set[(Int, Float)]] = new HashMap[Int,
    mutable.Set[(Int, Float)]]()

  private var firstGet: Boolean = true


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

  def addVertex(vId: Int): Unit = synchronized {
    srcVertexMap.put(vId, mutable.Set.empty[(Int, Float)])
  }

  def getNumVertices: Int = {
    srcVertexMap.size
  }

  def getNumEdges: Int = {
    srcVertexMap.values.foldLeft(0)(_ + _.size)
  }

  /**
    * The reset is mainly for the unit test purpose. It does not reset the size of data
    * structures that are initially set by calling setUp function.
    */
  def reset {
    srcVertexMap.clear()
  }

  def getNeighbors(vid: Int): mutable.Set[(Int, Float)] = {
    srcVertexMap.getOrElse(vid, mutable.Set.empty[(Int, Float)])
  }
}
