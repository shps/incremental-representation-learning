package au.csiro.data61.randomwalk.common

import java.io.FileInputStream

import spray.json._

import scala.collection.mutable
import scala.collection.parallel.ParSeq
import scala.io.Source

//import play.api.libs.json.Json

/**
  * Created by Hooman on 2018-02-27.
  */
object DatasetCleaner {

  def checkRedundantEdges(edges: Seq[(Int, Int)]) = {
    val rEdges = edges.flatMap { case (a, b) => Seq((a, b), (b, a)) }
    if (edges.size * 2 != rEdges.distinct.size) {
      throw new Exception("There are reversed edges.")
    }
  }

  def checkDataSet(config: Params, initId: Int): Unit = {
    val fm = FileManager(config)
    val edges = fm.readEdgeList()
    println(s"Number of edges: ${edges.size}")
    checkRedundantEdges(edges)
    val vertices = edges.flatMap { case e => Seq(e._1, e._2) }.distinct.sortWith(_ < _)
    println(s"Number of vertices: ${vertices.size}")
    for (i <- 0 until vertices.size) {
      val expected = initId + i
      if (vertices(i) != expected) {
        throw new Exception(s"Vertex ID ${vertices(i)} not equal the expected ID ${expected}")
      }
    }
  }

  def convertJsonFile(config: Params) {
    val fm = FileManager(config)
    val coAuthors = fm.readJsonFile()
    fm.saveCoAuthors(coAuthors.toList)
    val filtered = coAuthors.flatMap { case (src, dst, y) => Seq(src, dst) }
      .distinct.zipWithIndex
    fm.saveIds(filtered)

    //    val altNames = coAuthors.filter(_._3 == 0).map { case (n1, n2, y) => (n1, n2) }.groupBy
    // (_._1)
    //    val altKeys = altNames.keySet.seq
    //    val nameMap = new mutable.HashMap[String, String]()
    //    for (k <- altKeys) {
    //      val otherNames = altNames.getOrElse(k, ParSeq.empty)
    //      for (name <- otherNames) {
    //          nameMap.put(name, k)
    //      }
    //    }
    //    fm.saveCoAuthors(coAuthors.toList)
  }

}
