package au.csiro.data61.randomwalk.algorithm

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, HashMap}


/**
  *
  */

object GraphMap {

  private lazy val srcVertexMap: mutable.Map[Int, Seq[(Int, Float)]] = new HashMap[Int, Seq[(Int,
    Float)]]()
  //  private lazy val offsets: ArrayBuffer[Int] = new ArrayBuffer()
  //  private lazy val lengths: ArrayBuffer[Int] = new ArrayBuffer()
  //  private lazy val edges: ArrayBuffer[(Int, Float)] = new ArrayBuffer()
  //  private var indexCounter: Int = 0
  //  private var offsetCounter: Int = 0
  private var firstGet: Boolean = true


  def addVertex(vId: Int, neighbors: Seq[(Int, Float)]): Unit = synchronized {
    srcVertexMap.get(vId) match {
      case None => {
        srcVertexMap.put(vId, neighbors)
      }
      case Some(value) => value
    }
  }

  def putVertex(vId: Int, neighbors: Seq[(Int, Float)]): Unit = synchronized {
    srcVertexMap.put(vId, neighbors)
  }

  def getVertices(): Seq[Int] = {
    val vertices: Seq[Int] = srcVertexMap.keys.toSeq
    vertices
  }

  //  private def updateIndices(vId: Int, outDegree: Int): Unit = {
  //    srcVertexMap.put(vId, indexCounter)
  //    offsets.insert(indexCounter, offsetCounter)
  //    lengths.insert(indexCounter, outDegree)
  //    indexCounter += 1
  //  }

  def getGraphStatsOnlyOnce: (Int, Int) = synchronized {
    if (firstGet) {
      firstGet = false

      (srcVertexMap.size, getNumEdges)
    }
    else
      (0, 0)
  }

  def resetGetters {
    firstGet = true
  }

  def addVertex(vId: Int): Unit = synchronized {
    srcVertexMap.put(vId, ArrayBuffer.empty[(Int, Float)])
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
    resetGetters
    srcVertexMap.clear()
  }

  def getNeighbors(vid: Int): Seq[(Int, Float)] = {
    srcVertexMap.getOrElse(vid, Seq.empty[(Int, Float)])
  }
}
