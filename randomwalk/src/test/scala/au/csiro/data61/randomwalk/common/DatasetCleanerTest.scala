package au.csiro.data61.randomwalk.common

import java.time.Instant

import au.csiro.data61.randomwalk.algorithm.{GraphMap, UniformRandomWalk}
import au.csiro.data61.randomwalk.experiments.StreamingExperiment
import org.scalatest.FunSuite


/**
  * Created by Hooman on 2018-02-27.
  */
class DatasetCleanerTest extends FunSuite {

  private val dataset = "/Users/Ganymedian/Desktop/dynamic-rw/datasets/"

  test("testCheckDataSet") {
    val fName = dataset + "cocit_edgelist.txt"
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
    var fName = dataset + "cocit1_edgelist.txt"
    val output = dataset
    var config = Params(input = fName, output = output, delimiter = "\\s+", directed = false)
    UniformRandomWalk(config).loadGraph()
    val comp = DatasetCleaner.getBiggestSccAndCounts()
    println(s"Biggest SSC: ${comp._1.size}")

    //    var edges = Set.empty[(Int, Int)]
    //    for (v <- comp) {
    //      for (dst <- GraphMap.getNeighbors(v)) {
    //        var edge = (v, dst._1)
    //        if (dst._1 > v)
    //          edge = (dst._1, v)
    //        edges ++= Set(edge)
    //      }
    //    }
    //    val self = edges.filter(a => a._1 == a._2)
    //    println(s"Number of self edges: ${self.size}")
    //    edges = edges.filter(a => a._1 != a._2)
    //    println(s"Number of edges in the component: ${edges.size}")
    //
    //    val fm = FileManager(config)
    //
    //    fm.saveEdgeList(edges, "cocit1_edgelist")
    //
    //    val labels = fm.readLabels(dataset + "cocit_labels.txt")
    //
    //    val vSet = comp.toSet
    //    val compLabels = labels.filter(a => vSet.contains(a._1))
    //
    //    println(s"Number of vertices in the component labels: ${compLabels.size}")
    //    fm.saveLabels(compLabels.seq.toSet, "cocit1_labels")
  }

  test("Convert to undirected") {
    var fName = dataset + "cocit_edgelist.txt"
    val output = dataset
    var config = Params(input = fName, output = output, delimiter = "\\s+", directed = false)
    UniformRandomWalk(config).loadGraph()
    val undirectedEdges = DatasetCleaner.convertToUndirected()
    println(s"Number of undirected edges: ${undirectedEdges.size}")
    //    var edges = Set.empty[(Int, Int)]

    val self = undirectedEdges.filter(a => a._1 == a._2)
    println(s"Number of self edges: ${self.size}")
    val edges = undirectedEdges.filter(a => a._1 != a._2)
    println(s"Number of edges in the component: ${edges.size}")

    //    val fm = FileManager(config)
    //
    //    fm.saveEdgeList(edges.toSet.seq, "cocit_edgelist_undir")
    //
    //    val labels = fm.readLabels(dataset + "cora_labels.txt")
    //
    //    val vSet = comp.toSet
    //    val compLabels = labels.filter(a => vSet.contains(a._1))
    //
    //    println(s"Number of vertices in the component labels: ${compLabels.size}")
    //    fm.saveLabels(compLabels.seq.toSet, "cora1_labels")
  }

//  test("Draw density") {
//    val fName = dataset + "blog_edgelist.txt"
//    val outputFile = "blog-density"
//    val output = dataset
//    val config = Params(input = fName, output = output, delimiter = ",", directed = false,
//      edgeStreamSize = 50, initEdgeSize = 0.1f)
//    val fm = FileManager(config)
//    val (initEdges, edges) = fm.readPartitionEdgeListWithInitEdges()
//    val experiment = StreamingExperiment(config)
//    var densities = Seq.empty[Double]
//    experiment.updateGraph(initEdges)
//    densities ++= Seq(computeDensity())
//    for (stream <- edges) {
//      experiment.updateGraph(stream._2)
//      densities ++= Seq(computeDensity())
//    }
//
//    fm.saveDensities(densities, outputFile)
//  }

  def computeDensity(): Double = {
    val m = GraphMap.getNumEdges.toDouble // It is already multiplying by 2.
    val n = GraphMap.getNumVertices.toDouble

    val d: Double = m / (n * (n - 1))
    return d
  }
}
