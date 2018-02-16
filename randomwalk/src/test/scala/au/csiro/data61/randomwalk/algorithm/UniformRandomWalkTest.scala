package au.csiro.data61.randomwalk.algorithm

import au.csiro.data61.randomwalk.common.CommandParser.TaskName
import au.csiro.data61.randomwalk.common.{FileManager, Params}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.BeforeAndAfter

import scala.collection.mutable


class UniformRandomWalkTest extends org.scalatest.FunSuite with BeforeAndAfter {

  private val karate = "./src/test/resources/karate.txt"
  private val testGraph = "./src/test/resources/testgraph.txt"
  private val master = "local[*]" // Note that you need to verify unit tests in a multi-core
  // computer.
  private val appName = "rw-unit-test"
  private var sc: SparkContext = _

  before {
    // Note that the Unit Test may throw "java.lang.AssertionError: assertion failed: Expected
    // hostname"
    // If this test is running in MacOS and without Internet connection.
    // https://issues.apache.org/jira/browse/SPARK-19394
    val conf = new SparkConf().setMaster(master).setAppName(appName)
    sc = SparkContext.getOrCreate(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
    GraphMap.reset
  }

  test("load graph as undirected") {
    val config = Params(input = karate, directed = false, numWalks = 1)
    val rw = UniformRandomWalk(sc, config)
    val paths = rw.loadGraph() // loadGraph(int)
    assert(rw.nEdges == 156)
    assert(rw.nVertices == 34)
    assert(paths.count() == 34)
    val vAcc = sc.longAccumulator("v")
    val eAcc = sc.longAccumulator("e")
    paths.coalesce(1).mapPartitions { iter =>
      vAcc.add(GraphMap.getNumVertices)
      eAcc.add(GraphMap.getNumEdges)
      iter
    }.first()
    assert(eAcc.sum == 156)
    assert(vAcc.sum == 34)
  }

  test("load graph as directed") {
    val config = Params(input = karate, directed = true, numWalks = 1)
    val rw = UniformRandomWalk(sc, config)
    val paths = rw.loadGraph()
    assert(rw.nEdges == 78)
    assert(rw.nVertices == 34)
    assert(paths.count() == 34)
    val vAcc = sc.longAccumulator("v")
    val eAcc = sc.longAccumulator("e")
    paths.coalesce(1).mapPartitions { iter =>
      vAcc.add(GraphMap.getNumVertices)
      eAcc.add(GraphMap.getNumEdges)
      iter
    }.first()
    assert(eAcc.sum == 78)
    assert(vAcc.sum == 34)
  }

  test("remove vertex") {
    val config = Params(input = karate, directed = false)
    val rw = UniformRandomWalk(sc, config)
    val before = FileManager(sc, config).readFromFile()
    for (target <- 1 until 34) {
      rw.buildGraphMap(before)
      val neighbors = GraphMap.getNeighbors(target)
      val nDegrees = new mutable.HashMap[Int, Int]()
      for (n <- neighbors) {
        nDegrees.put(n._1, GraphMap.getNeighbors(n._1).length)
      }

      val after = Experiments(sc, config).removeVertex(before, target)
      assert(after.count() == 33)
      assert(after.filter(_._1 == target).count() == 0)
      rw.buildGraphMap(after)
      for (n <- neighbors) {
        val dstNeighbors = GraphMap.getNeighbors(n._1)
        assert(!dstNeighbors.map(_._1).contains(target))
        val newDegree = dstNeighbors.length
        nDegrees.get(n._1) match {
          case Some(d) => assert(d - 1 == newDegree)
          case None => assert(false)
        }

      }
    }
  }

  test("graph map load after removal") {
    val config = Params(input = karate, directed = false)
    val rw = UniformRandomWalk(sc, config)
    val exp = Experiments(sc, config)
    val before = FileManager(sc, config).readFromFile()
    for (target <- 1 until 34) {
      rw.buildGraphMap(before)
      val neighbors = GraphMap.getNeighbors(target)
      val nDegrees = new mutable.HashMap[Int, Int]()
      for (n <- neighbors) {
        nDegrees.put(n._1, GraphMap.getNeighbors(n._1).length)
      }

      val after = exp.removeVertex(before, target)
      rw.buildGraphMap(after)
      rw.buildGraphMap(before)
      val neighbors2 = GraphMap.getNeighbors(target)
      val nDegrees2 = new mutable.HashMap[Int, Int]()
      for (n <- neighbors2) {
        nDegrees2.put(n._1, GraphMap.getNeighbors(n._1).length)
      }
      assert(neighbors2 sameElements neighbors)
      assert(nDegrees2 sameElements nDegrees)
    }
  }

  test("filter affected paths") {
    val p1 = Array(1, 2, 1, 2)
    val p2 = Array(3, 4, 3, 4)
    val p3 = Array(5, 6, 5, 6)
    val afs = Array(1, 5, 6)
    val pRdd = sc.parallelize(Array(p1, p2, p3))
    val config = Params()
    val rw = Experiments(sc, config)
    val result = rw.filterAffectedPaths(pRdd, afs).collect()
    assert(result.size == 2)
    assert(result(0) sameElements p1)
    assert(result(1) sameElements p3)
  }

  test("filter split paths") {
    val p1 = Array(1, 2, 3, 4)
    val p2 = Array(3, 4, 3, 4)
    val p3 = Array(4, 6, 5, 6)
    val p4 = Array(4, 3, 5, 6)
    val afs = Array(1, 5, 6)
    val pRdd = sc.parallelize(Array(p1, p2, p3, p4))
    val config = Params()
    val rw = Experiments(sc, config)
    val result = rw.filterSplitPaths(pRdd, afs).collect()
    assert(result.size == 3)
    val vertices = result.map(_._1)
    assert(vertices sameElements Array(1, 4, 4))
    val paths = result.map(_._2)
    assert(paths(0) sameElements Array(1))
    assert(paths(1) sameElements Array(4, 6))
    assert(paths(2) sameElements Array(4, 3, 5))
  }

  test("init walker") {
    val config = Params(numWalks = 5)
    val rw = UniformRandomWalk(sc, config)
    val v = 1
    val walkers = rw.initWalker(v).collect()
    assert(walkers.length == config.numWalks)
    assert(walkers.forall { case (a, b) => a == v && (b sameElements Array(v)) })
  }

  test("first order walk") {
    // Undirected graph
    val wLength = 50

    val config = Params(input = karate, directed = false, walkLength =
      wLength, rddPartitions = 8, numWalks = 2)
    val rw = UniformRandomWalk(sc, config)
    val rValue = 0.1f
    val nextFloatGen = () => rValue
    val graph = rw.loadGraph()
    val paths = rw.firstOrderWalk(graph, nextFloatGen)
    assert(paths.count() == 2 * rw.nVertices) // a path per vertex
    val rw2 = UniformRandomWalk(sc, config)
    val gMap = FileManager(sc, config).readFromFile().collectAsMap()
    val rSampler = RandomSample(nextFloatGen)
    paths.collect().foreach { case (p: Array[Int]) =>
      val p2 = doFirstOrderRandomWalk(gMap, p(0), wLength, rSampler)
      assert(p sameElements p2)
    }
  }

  //  test("addAndRun m2") {
  //    // Undirected graph
  //    val wLength = 5
  //
  //    val config = Params(input = karate, directed = false, walkLength =
  //      wLength, rddPartitions = 8, numWalks = 2, rrType = RrType.m2)
  //    val rw = UniformRandomWalk(sc, config)
  //    rw.addAndRun()
  //  }
  //
  //  test("addAndRun m3") {
  //    // Undirected graph
  //    val wLength = 5
  //
  //    val config = Params(input = karate, directed = false, walkLength =
  //      wLength, rddPartitions = 8, numWalks = 1, rrType = RrType.m3)
  //    val rw = UniformRandomWalk(sc, config)
  //    rw.addAndRun()
  //  }
  //
  //  test("addAndRun m4") {
  //    // Undirected graph
  //    val wLength = 100
  //
  //    val config = Params(input = karate, directed = false, walkLength =
  //      wLength, rddPartitions = 8, numWalks = 100, rrType = RrType.m4)
  //    val rw = UniformRandomWalk(sc, config)
  //    rw.addAndRun()
  //  }

  test("Query Nodes") {
    var config = Params(nodes = "1 2 3 4")

    val p1 = Array(1, 2, 1, 2, 1)
    val p2 = Array(2, 2, 2, 2, 1)
    val p3 = Array(3, 4, 2, 3)
    val p4 = Array(4)
    val expected = Array((1, (4, 2)), (2, (7, 3)), (3, (2, 1)), (4, (2, 2)))

    val paths = sc.parallelize(Array(p1, p2, p3, p4))
    var rw = UniformRandomWalk(sc, config)
    var counts = rw.queryPaths(paths)
    assert(counts sameElements expected)

    config = Params()
    rw = UniformRandomWalk(sc, config)
    counts = rw.queryPaths(paths)
    assert(counts sameElements expected)
  }

  test("Experiments") {
    val query = 1 to 34 toArray
    var config = Params(input = karate,
      output = "", directed = false, walkLength = 10,
      rddPartitions = 8, numWalks = 1, cmd = TaskName.firstorder, nodes = query.mkString(" "))
    var rw = UniformRandomWalk(sc, config)
    val g = rw.loadGraph()
    val paths = rw.firstOrderWalk(g)
    val counts1 = rw.queryPaths(paths)
    assert(counts1.length == 34)

    config = Params(input = karate, directed = false, walkLength = 10,
      rddPartitions = 8, numWalks = 1, cmd = TaskName.firstorder)

    rw = UniformRandomWalk(sc, config)
    val counts2 = rw.queryPaths(paths)
    assert(counts2.length == 34)

    assert(counts1.sortBy(_._1) sameElements counts2.sortBy(_._1))
  }

  private def doFirstOrderRandomWalk(gMap: scala.collection.Map[Int, Array[(Int, Float)]], src: Int,
                                     walkLength: Int, rSampler: RandomSample): Array[Int]
  = {
    var path = Array(src)

    for (_ <- 0 until walkLength) {

      val curr = path.last
      val currNeighbors = gMap.get(curr) match {
        case Some(neighbors) => neighbors
        case None => Array.empty[(Int, Float)]
      }
      if (currNeighbors.size > 0) {
        path = path ++ Array(rSampler.sample(currNeighbors)._1)
      } else {
        return path
      }
    }

    path
  }

  test("analyze")
  {
    val config = Params(input=karate, directed = false, rddPartitions = 10)
    val ea = ExperimentAnalyzer(sc, config)
    val rw = UniformRandomWalk(sc, config)
    rw.loadGraph()
    ea.analyze(rw.degrees())
  }
}
