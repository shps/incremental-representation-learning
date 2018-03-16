package au.csiro.data61.randomwalk.algorithm

import au.csiro.data61.randomwalk.common.CommandParser.{RrType, TaskName, WalkType}
import au.csiro.data61.randomwalk.common.{FileManager, Params}
import org.scalatest.BeforeAndAfter

import scala.collection.mutable
import scala.collection.parallel.ParSeq


class UniformRandomWalkTest extends org.scalatest.FunSuite with BeforeAndAfter {

  private val karate = "./src/test/resources/karate.txt"
  private val testGraph = "./src/test/resources/testgraph.txt"


  after {
    GraphMap.reset
  }

  test("load graph as undirected") {
    val config = Params(input = karate, directed = false, numWalks = 1)
    val rw = UniformRandomWalk(config)
    val paths = rw.loadGraph() // loadGraph(int)
    assert(rw.nEdges == 156)
    assert(rw.nVertices == 34)
    assert(paths.length == 34)

    assert(GraphMap.getNumEdges == 156)
    assert(GraphMap.getNumVertices == 34)
  }

  test("load graph as directed") {
    val config = Params(input = karate, directed = true, numWalks = 1)
    val rw = UniformRandomWalk(config)
    val paths = rw.loadGraph()
    assert(rw.nEdges == 78)
    assert(rw.nVertices == 34)
    assert(paths.length == 34)
    assert(GraphMap.getNumEdges == 78)
    assert(GraphMap.getNumVertices == 34)
  }

  test("remove vertex") {
    val config = Params(input = karate, directed = false)
    val rw = UniformRandomWalk(config)
    val before = FileManager(config).readFromFile(config.directed)
    for (target <- 1 until 34) {
      rw.buildGraphMap(before.seq)
      val neighbors = GraphMap.getNeighbors(target)
      val nDegrees = new mutable.HashMap[Int, Int]()
      for (n <- neighbors) {
        nDegrees.put(n._1, GraphMap.getNeighbors(n._1).length)
      }

      val after = Experiments(config).removeVertex(before, target)
      assert(after.length == 33)
      assert(after.filter(_._1 == target).length == 0)
      rw.buildGraphMap(after.seq)
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
    val rw = UniformRandomWalk(config)
    val exp = Experiments(config)
    val before = FileManager(config).readFromFile(config.directed)
    for (target <- 1 until 34) {
      rw.buildGraphMap(before.seq)
      val neighbors = GraphMap.getNeighbors(target)
      val nDegrees = new mutable.HashMap[Int, Int]()
      for (n <- neighbors) {
        nDegrees.put(n._1, GraphMap.getNeighbors(n._1).length)
      }

      val after = exp.removeVertex(before, target)
      rw.buildGraphMap(before.seq)
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
    val p1 = Seq(1, 2, 1, 2)
    val p2 = Seq(3, 4, 3, 4)
    val p3 = Seq(5, 6, 5, 6)
    val afs = Seq(1, 5, 6)
    val pRdd = ParSeq(p1, p2, p3)
    val config = Params()
    val rw = Experiments(config)
    val result = rw.filterAffectedPaths(pRdd, afs)
    assert(result.size == 2)
    assert(result(0) sameElements p1)
    assert(result(1) sameElements p3)
  }

  test("filter split paths") {
    val p1 = Seq(1, 2, 3, 4)
    val p2 = Seq(3, 4, 3, 4)
    val p3 = Seq(4, 6, 5, 6)
    val p4 = Seq(4, 3, 5, 6)
    val afs = Seq(1, 5, 6)
    val pRdd = ParSeq(p1, p2, p3, p4)
    val config = Params()
    val rw = Experiments(config)
    val result = rw.filterSplitPaths(pRdd, afs)
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
    val rw = UniformRandomWalk(config)
    val v = 1
    val walkers = rw.initWalker(v)
    assert(walkers.length == config.numWalks)
    assert(walkers.forall { case (a, b) => a == v && (b sameElements Array(v)) })
  }

  test("first order walk") {
    // Undirected graph
    val wLength = 50

    val config = Params(input = karate, directed = false, walkLength =
      wLength, rddPartitions = 8, numWalks = 2)
    val rw = UniformRandomWalk(config)
    val rValue = 0.1f
    val nextFloatGen = () => rValue
    val graph = rw.loadGraph()
    val paths = rw.firstOrderWalk(graph, nextFloatGen)
    assert(paths.length == 2 * rw.nVertices) // a path per vertex
    val rw2 = UniformRandomWalk(config)
    val gMap = FileManager(config).readFromFile(config.directed).groupBy(_._1).map {
      case (k, v) => (k, v.flatMap(_._2).seq)
    }
    val rSampler = RandomSample(nextFloatGen)
    paths.seq.foreach { case (p: Seq[Int]) =>
      val p2 = doFirstOrderRandomWalk(gMap.seq, p(0), wLength, rSampler)
      assert(p sameElements p2)
    }
  }

  test("test 2nd order random walk undirected3") {
    // Undirected graph
    val wLength = 50
    val config = Params(input = karate, directed = false, walkLength =
      wLength, rddPartitions = 8, numWalks = 1)
    val rValue = 0.9f
    val nextFloatGen = () => rValue
    val rw = UniformRandomWalk(config)
    val graph = rw.loadGraph()
    val paths = rw.secondOrderWalk(graph, nextFloatGen)
    assert(paths.length == rw.nVertices) // a path per vertex
    val rSampler = RandomSample(nextFloatGen)
    val gMap = FileManager(config).readFromFile(config.directed).groupBy(_._1).map {
      case (k, v) => (k, v.flatMap(_._2).seq)
    }
    paths.seq.foreach { case (p: Seq[Int]) =>
      val p2 = doSecondOrderRandomWalk(gMap.seq, p(0), wLength, rSampler, 1.0f, 1.0f)
      assert(p sameElements p2)
    }
  }

  test("test 2nd order random walk undirected2") {
    // Undirected graph
    val wLength = 50
    val config = Params(input = karate, directed = false, walkLength =
      wLength, rddPartitions = 8, numWalks = 1, p = 0.5f, q = 2.0f)
    val rValue = 0.1f
    val nextFloatGen = () => rValue
    val rw = UniformRandomWalk(config)
    val graph = rw.loadGraph()
    val paths = rw.secondOrderWalk(graph, nextFloatGen)
    assert(paths.length == rw.nVertices) // a path per vertex
    val rSampler = RandomSample(nextFloatGen)
    val gMap = FileManager(config).readFromFile(config.directed).groupBy(_._1).map {
      case (k, v) => (k, v.flatMap(_._2).seq)
    }
    paths.seq.foreach { case (p: Seq[Int]) =>
      val p2 = doSecondOrderRandomWalk(gMap.seq, p(0), wLength, rSampler, p = config.p, q = config.q)
      assert(p sameElements p2)
    }
  }

    test("streaming updates") {
      // Undirected graph
      val wLength = 4

      val config = Params(input = karate, directed = false, walkLength =
        wLength, rddPartitions = 8, numWalks = 1, rrType = RrType.m3, numVertices = 34, wType = WalkType.secondorder)

      val exp = Experiments(config)
      exp.streamingUpdates()

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

  //  test("Query Nodes") {
  //    var config = Params(nodes = "1 2 3 4")
  //
  //    val p1 = Array(1, 2, 1, 2, 1)
  //    val p2 = Array(2, 2, 2, 2, 1)
  //    val p3 = Array(3, 4, 2, 3)
  //    val p4 = Array(4)
  //    val expected = Array((1, (4, 2)), (2, (7, 3)), (3, (2, 1)), (4, (2, 2)))
  //
  //    val paths = Array(p1, p2, p3, p4)
  //    var rw = UniformRandomWalk(config)
  //    var counts = rw.queryPaths(paths)
  //    assert(counts sameElements expected)
  //
  //    config = Params()
  //    rw = UniformRandomWalk(config)
  //    counts = rw.queryPaths(paths)
  //    assert(counts sameElements expected)
  //  }

  test("Experiments") {
    val query = 1 to 34 toArray
    var config = Params(input = karate,
      output = "", directed = false, walkLength = 10,
      rddPartitions = 8, numWalks = 1, cmd = TaskName.firstorder, nodes = query.mkString(" "))
    var rw = UniformRandomWalk(config)
    val g = rw.loadGraph()
    val paths = rw.firstOrderWalk(g)
    val counts1 = rw.queryPaths(paths.seq)
    assert(counts1.length == 34)

    config = Params(input = karate, directed = false, walkLength = 10,
      rddPartitions = 8, numWalks = 1, cmd = TaskName.firstorder)

    rw = UniformRandomWalk(config)
    val counts2 = rw.queryPaths(paths.seq)
    assert(counts2.length == 34)

    assert(counts1.seq.sortBy(_._1) sameElements counts2.seq.sortBy(_._1))
  }

  private def doFirstOrderRandomWalk(gMap: Map[Int, Seq[(Int, Float)]], src: Int,
                                     walkLength: Int, rSampler: RandomSample): Array[Int]
  = {
    var path = Array(src)

    for (_ <- 0 until walkLength) {

      val curr = path.last
      val currNeighbors = gMap.get(curr) match {
        case Some(neighbors) => neighbors
        case None => Seq.empty[(Int, Float)]
      }
      if (currNeighbors.size > 0) {
        path = path ++ Array(rSampler.sample(currNeighbors)._1)
      } else {
        return path
      }
    }

    path
  }

  private def doSecondOrderRandomWalk(gMap: Map[Int, Seq[(Int, Float)]], src: Int,
                                      walkLength: Int, rSampler: RandomSample, p: Float,
                                      q: Float): Seq[Int]
  = {
    var path = Array(src)
    val neighbors = gMap.get(src) match {
      case Some(neighbors) => neighbors
      case None => Seq.empty[(Int, Float)]
    }
    if (neighbors.length > 0) {
      path = path ++ Array(rSampler.sample(neighbors)._1)
    }
    else {
      return path
    }

    for (_ <- 0 until walkLength) {

      val curr = path.last
      val prev = path(path.length - 2)
      val currNeighbors = gMap.get(curr) match {
        case Some(neighbors) => neighbors
        case None => Seq.empty[(Int, Float)]
      }
      if (currNeighbors.length > 0) {
        val prevNeighbors = gMap.get(prev) match {
          case Some(neighbors) => neighbors
          case None => Seq.empty[(Int, Float)]
        }
        path = path ++ Array(rSampler.secondOrderSample(p, q, prev, prevNeighbors, currNeighbors)
          ._1)
      } else {
        return path
      }
    }

    path
  }

  //  test("analyze") {
  //    val config = Params(input = karate, directed = false, rddPartitions = 10)
  //    val ea = ExperimentAnalyzer(config)
  //    val rw = UniformRandomWalk(config)
  //    rw.loadGraph()
  //    ea.analyze(rw.degrees())
  //  }
}
