package au.csiro.data61.randomwalk.common

import java.time.Instant

import au.csiro.data61.randomwalk.algorithm.{GraphMap, UniformRandomWalk}
import org.scalatest.FunSuite


/**
  * Created by Hooman on 2018-02-27.
  */
class DatasetCleanerTest extends FunSuite {

  private val dataset = "/Users/Ganymedian/Desktop/dynamic-rw/datasets/"

  test("testCheckDataSet") {
    val fName = dataset + "cora_edgelist.txt"
    val initId = 0
    val config = Params(input = fName, delimiter = "\\s+", directed = true)
    DatasetCleaner.checkDataSet(config, initId)

  }

  case class CoAuthor(a1: String, a2: String, year: Int)

  test("jsonConvertor") {
    val fName = dataset + "test.json"
    val output = dataset
    val config = Params(input = fName, output = output)
    DatasetCleaner.convertJsonFile(config)

  }

  test("test") {
    val fName = dataset + "ia-enron-employees.txt"
    val config = Params(input = fName, delimiter = "\\s+", directed = false)
    val timedEdges = FileManager(config).readTimestampedEdgeList().seq.sortWith(_._3 > _._3).par
    val times = timedEdges.map(a => (a._1, a._2, Instant.ofEpochMilli(a._3)))
    System.out
  }

  test("SCCs") {
    var fName = dataset + "cora1_edgelist.txt"
    var config = Params(input = fName, delimiter = "\\s+", directed = false)
    UniformRandomWalk(config).loadGraph()
    var components = DatasetCleaner.countNumberOfSCCs()
    println(s"Number of SSCs in Cora: ${components}")

    //    GraphMap.reset
    //    fName = dataset + "Wiki_edgelist.txt"
    //    config = Params(input = fName, delimiter = "\\s+", directed = false)
    //    UniformRandomWalk(config).loadGraph()
    //    components = DatasetCleaner.countNumberOfSCCs()
    //    println(s"Number of SSCs in Wiki: $components")
    //
    //    GraphMap.reset
    //    fName = dataset + "edges.txt"
    //    config = Params(input = fName, delimiter = ",", directed = false)
    //    UniformRandomWalk(config).loadGraph()
    //    components = DatasetCleaner.countNumberOfSCCs()
    //    println(s"Number of SSCs in BlogCatalog: $components")
  }

  test("Biggest Scc") {
    var fName = dataset + "cora_edgelist.txt"
    val output = dataset
    var config = Params(input = fName, output = output, delimiter = "\\s+", directed = false)
    UniformRandomWalk(config).loadGraph()
    val comp = DatasetCleaner.getBiggestScc()
    println(s"Biggest SSC: ${comp.size}")

    var edges = Set.empty[(Int, Int)]
    for (v <- comp) {
      for (dst <- GraphMap.getNeighbors(v)) {
        var edge = (v, dst._1)
        if (dst._1 > v)
          edge = (dst._1, v)
        edges ++= Set(edge)
      }
    }
    val self = edges.filter(a => a._1 == a._2)
    println(s"Number of self edges: ${self.size}")
    edges = edges.filter(a => a._1 != a._2)
    println(s"Number of edges in the component: ${edges.size}")

    val fm = FileManager(config)

    fm.saveEdgeList(edges, "cora1_edgelist")

    val labels = fm.readLabels(dataset + "cora_labels.txt")

    val vSet = comp.toSet
    val compLabels = labels.filter(a => vSet.contains(a._1))

    println(s"Number of vertices in the component labels: ${compLabels.size}")
    fm.saveLabels(compLabels.seq.toSet, "cora1_labels")
  }
}
