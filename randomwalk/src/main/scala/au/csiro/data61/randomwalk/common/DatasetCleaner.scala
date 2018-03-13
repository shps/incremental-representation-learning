package au.csiro.data61.randomwalk.common

import scala.collection.parallel.ParSeq


/**
  * Created by Hooman on 2018-02-27.
  */
object DatasetCleaner {

  def checkRedundantEdges(edges: ParSeq[(Int, Int)]) = {
    val rEdges = edges.flatMap { case (a, b) => Seq((a, b), (b, a)) }.distinct
    if (edges.size * 2 != rEdges.size) {
      throw new Exception(s"There are reversed edges. Expected: ${edges.size * 2}, Actual: " +
        s"${rEdges.size}")
    }
  }

  def checkDataSet(config: Params, initId: Int): Unit = {
    val fm = FileManager(config)
    val edges = fm.readEdgeList()
    println(s"Number of edges: ${edges.size}")
    val deduplicated = edges.distinct
    println(s"Number of edges after deduplication: ${deduplicated.size}")
    checkRedundantEdges(edges)
    val vertices = edges.flatMap { case e => Seq(e._1, e._2) }.distinct.seq.sortWith(_ < _)
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
    val authors = coAuthors.flatMap { case (src, dst, y) => Seq(src, dst) }
      .distinct.zipWithIndex
    fm.saveIds(authors)
    val idMaps = authors.toMap
    // saves as indirected, removes duplicates.
    val filtered = coAuthors.filter(_._3 != 0).map { case (a1, a2, y) =>
      val src = idMaps.getOrElse(a1, throw new Exception)
      val dst = idMaps.getOrElse(a2, throw new Exception)
      if (src < dst)
        (src, dst, y)
      else
        (dst, src, y)
    }.seq.sortWith(_._3 < _._3).par.distinct
    fm.saveCoAuthors(filtered)
    val groups = filtered.groupBy(_._3)
    val sums = groups.map { case (y, edges) =>
      (y, edges.length)
    }.toSeq.seq.sortWith(_._1 < _._1).map { case (y, n) => s"$y\t$n" }.mkString("\n")
    fm.saveNumAuthors(sums, "coauthors-per-year")

    val ua = groups.map { case (y, edges) =>
      val count = edges.flatMap { case (src, dst, _) => Seq(src, dst) }.distinct.length
      (y, count)
    }.toSeq.seq.sortWith(_._1 < _._1).map { case (y, n) => s"$y\t$n" }.mkString("\n")
    fm.saveNumAuthors(ua, "unique-authors-per-year")

//    val ra = groups.map { case (y, edges) =>
//      val count = edges.flatMap { case (src, dst, _) => Seq((src, dst),(dst, src)) }.distinct.length
//      (y, edges.length*2 - count)
//    }.toSeq.seq.sortWith(_._1 < _._1).map { case (y, n) => s"$y\t$n" }.mkString("\n")
//    println(ra)
  }

}
