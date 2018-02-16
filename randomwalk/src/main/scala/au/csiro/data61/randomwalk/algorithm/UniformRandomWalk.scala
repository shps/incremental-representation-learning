package au.csiro.data61.randomwalk.algorithm

import java.util

import au.csiro.data61.randomwalk.common.{FileManager, Params}
import org.apache.log4j.LogManager
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{HashPartitioner, SparkContext}

import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

case class UniformRandomWalk(context: SparkContext, config: Params) extends Serializable {

  def computeAffecteds(vertices: RDD[Int], affectedLength: Int): RDD[(Int,
    Array[Int])] = {
    val bcL = context.broadcast(affectedLength)

    vertices.map { v =>
      def computeAffecteds(afs: Array[Int], visited: util.HashSet[Int], v: Int, al: Int,
                           length: Int)
      : Unit = {
        if (length >= al)
          return
        visited.add(v)
        val neighbors = GraphMap.getNeighbors(v)
        if (neighbors != null) {
          for (n <- neighbors) {
            if (!visited.contains(n._1)) {
              afs(length) += 1
              visited.add(n._1)
              computeAffecteds(afs, visited, n._1, al, length + 1)
            }
          }
        }
      }

      val affecteds = new Array[Int](bcL.value)
      val visited = new util.HashSet[Int]()
      visited.add(v)
      affecteds(0) = 1
      computeAffecteds(affecteds, visited, v, bcL.value, 0)

      (v, affecteds)
    }.sortBy(_._2.last, ascending = false)
  }

  def degrees(): Array[(Int, Int)] = {
    val vertices = GraphMap.getVertices()
    val n = vertices.length
    val degs = new Array[(Int, Int)](n)
    for (i <- 0 until n) {
      degs(i) = (vertices(i), GraphMap.getNeighbors(vertices(i)).length)
    }
    degs
  }


  def computeProbs(paths: RDD[Array[Int]]): Array[Array[Double]] = {
    val n = GraphMap.getVertices().length
    val matrix = Array.ofDim[Double](n, n)
    paths.collect().foreach { case p =>
      for (i <- 0 until p.length - 1) {
        matrix(p(i) - 1)(p(i + 1) - 1) += 1
      }
    }

    matrix.map { row =>
      val sum = row.sum
      row.map { o =>
        o / sum.toDouble
      }
    }
  }


  lazy val partitioner: HashPartitioner = new HashPartitioner(config.rddPartitions)
  lazy val logger = LogManager.getLogger("rwLogger")
  var nVertices: Int = 0
  var nEdges: Int = 0

  def execute(): RDD[Array[Int]] = {
    firstOrderWalk(loadGraph())
  }

  /**
    * Loads the graph and computes the probabilities to go from each vertex to its neighbors
    *
    * @return
    */
  def loadGraph(): RDD[(Int, Array[Int])] = {

    val g: RDD[(Int, Array[(Int, Float)])] = FileManager(context, config).readFromFile()
    initRandomWalk(g)
  }

  def checkGraphMap() = {
    //    save(degrees())
    println(degrees().sortBy(_._1).map { case (v, d) => s"$v\t$d" }.mkString("\n"))
    for (v <- GraphMap.getVertices().sortBy(a => a)) {
      val n = GraphMap.getNeighbors(v).map(_._1)
      println(s"$v -> ${n.mkString(" ")}")
    }
  }

  def initWalker(v: Int): RDD[(Int, Array[Int])] = {
    val targetWalker = Array.fill(config.numWalks)(Array((v, Array(v)))).flatMap(a => a)
    context.parallelize(targetWalker)
  }


  def initRandomWalk(g: RDD[(Int, Array[(Int, Float)])]): RDD[(Int, Array[Int])] = {
    buildGraphMap(g)

    val vAccum = context.longAccumulator("vertices")
    val eAccum = context.longAccumulator("edges")

    g.foreachPartition { iter =>
      iter.foreach {
        case (_, (neighbors: Array[(Int, Float)])) =>
          vAccum.add(1)
          eAccum.add(neighbors.length)
      }
    }
    nVertices = vAccum.sum.toInt
    nEdges = eAccum.sum.toInt

    logger.info(s"edges: $nEdges")
    logger.info(s"vertices: $nVertices")
    println(s"edges: $nEdges")
    println(s"vertices: $nVertices")

    createWalkers(g)
  }

  def createWalkers(g: RDD[(Int, Array[(Int, Float)])]): RDD[(Int, Array[Int])] = {
    val walkers = g.mapPartitions({ iter =>
      iter.map {
        case (vId: Int, _) =>
          (vId, Array(vId))
      }
    }, preservesPartitioning = true
    )
    context.union(Array.fill(config.numWalks)(walkers))
  }

  def firstOrderWalk(initPaths: RDD[(Int, Array[Int])], nextFloat: () => Float = Random
    .nextFloat): RDD[Array[Int]] = {
    val walkLength = context.broadcast(config.walkLength)
    //    var totalPaths: RDD[Array[Int]] = context.emptyRDD[Array[Int]]

    //    for (_ <- 0 until config.numWalks) {
    val paths = initPaths.mapPartitions({ iter =>
      iter.map { case (_, steps) =>
        var path = steps
        val rSample = RandomSample(nextFloat)
        breakable {
          while (path.length < walkLength.value + 1) {
            val neighbors = GraphMap.getNeighbors(path.last)
            if (neighbors != null && neighbors.length > 0) {
              val (nextStep, _) = rSample.sample(neighbors)
              path = path ++ Array(nextStep)
            } else {
              break
            }
          }
        }
        path
      }
    }, preservesPartitioning = true
    ).persist(StorageLevel.MEMORY_AND_DISK)

    val pCount = paths.count()
    //    if (pCount != config.numWalks * nVertices) {
    //      println(s"Inconsistent number of paths: nPaths=[${pCount}] != vertices[$nVertices]")
    //    }
    //    totalPaths = totalPaths.union(paths).persist(StorageLevel
    //      .MEMORY_AND_DISK)
    //
    //    totalPaths.count()

    //    }

    //    totalPaths
    paths
  }

  def buildGraphMap(graph: RDD[(Int, Array[(Int, Float)])]): Unit = {
    GraphMap.reset // This is only to run on a single executor.
    graph.foreachPartition { iter: Iterator[(Int, Array[(Int, Float)])] =>
      iter.foreach { case (vId, neighbors) =>
        GraphMap.addVertex(vId, neighbors)
      }
    }

  }

  def queryPaths(paths: RDD[Array[Int]]): Array[(Int, (Int, Int))] = {
    var nodes: Array[Int] = Array.empty[Int]
    var numOccurrences: Array[(Int, (Int, Int))] = null
    if (config.nodes.isEmpty) {
      numOccurrences = paths.mapPartitions { iter =>
        iter.flatMap { case steps =>
          steps.groupBy(a => a).map { case (a, occurs) => (a, (occurs.length, 1)) }
        }
      }.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).collect()
    } else {
      nodes = config.nodes.split("\\s+").map(s => s.toInt)
      numOccurrences = new Array[(Int, (Int, Int))](nodes.length)

      for (i <- 0 until nodes.length) {
        val bcNode = context.broadcast(nodes(i))
        numOccurrences(i) = (nodes(i),
          paths.mapPartitions { iter =>
            val targetNode = bcNode.value
            iter.map { case steps =>
              val counts = steps.count(s => s == targetNode)
              val occurs = if (counts > 0) 1 else 0
              (counts, occurs)
            }
          }.reduce((c, o) => (c._1 + o._1, c._2 + o._2)))
      }
    }

    numOccurrences
  }

}
