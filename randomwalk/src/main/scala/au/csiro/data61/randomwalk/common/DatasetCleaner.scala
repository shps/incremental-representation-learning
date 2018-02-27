package au.csiro.data61.randomwalk.common

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

}
