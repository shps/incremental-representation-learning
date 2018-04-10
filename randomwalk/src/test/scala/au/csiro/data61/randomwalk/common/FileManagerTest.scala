package au.csiro.data61.randomwalk.common

import org.scalatest.FunSuite

/**
  * Created by Hooman on 2018-04-10.
  */
class FileManagerTest extends FunSuite {

  test("testReadPartitionedEdgeList") {
    val config = Params(input = "/Users/Ganymedian/Desktop/dynamic-rw/datasets/BlogCatalog-dataset/data/edges.txt", delimiter = ",", seed = 1234)
    val fm = FileManager(config)
    val edges = fm.readPartitionedEdgeList()
    println(edges)
  }

}
