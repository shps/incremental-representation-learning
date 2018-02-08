package au.csiro.data61.randomwalk.algorithm

import java.io.{BufferedWriter, File, FileWriter}
import java.util

import au.csiro.data61.randomwalk.common.CommandParser.RrType
import au.csiro.data61.randomwalk.common.{Params, Property}
import org.apache.log4j.LogManager
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{HashPartitioner, SparkContext}

import scala.util.control.Breaks.{break, breakable}
import scala.util.{Random, Try}

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

  def save(degrees: Array[Int]) = {
    val file = new File(s"${config.output}.${Property.degreeSuffix}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(degrees.zipWithIndex.map { case (d, i) => s"${i + 1}\t$d" }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveAffecteds(afs: RDD[(Int, Array[Int])]) = {
    afs.map {
      case (vId, af) =>
        s"$vId\t${af.mkString("\t")}"
    }.repartition(1).saveAsTextFile(s"${config.output}.${Property.affecteds}")
  }

  def degrees(): Array[Int] = {
    val degrees = new Array[Int](nVertices)
    GraphMap.getVertices().foreach(v => degrees(v - 1) = GraphMap.getNeighbors(v).length)
    degrees
  }

  def save(probs: Array[Array[Double]]): Unit = {
    val file = new File(s"${config.output}.${Property.probSuffix}.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(probs.map(array => array.map(a => f"$a%1.4f").mkString("\t")).mkString("\n"))
    bw.flush()
    bw.close()
  }


  def computeProbs(paths: RDD[Array[Int]]): Array[Array[Double]] = {
    val matrix = Array.ofDim[Double](nVertices, nVertices)
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

    val g: RDD[(Int, Array[(Int, Float)])] = readFromFile(config)
    initRandomWalk(g)
  }

  def removeAndRun(): Unit = {
    val g1 = readFromFile(config)
    val vertices: Array[Int] = g1.map(_._1).collect()
    for (target <- vertices) {
      logger.info(s"Removed vertex $target")
      println(s"Removed vertex $target")
      val g2 = removeVertex(g1, target)
      val init = initRandomWalk(g2)
      for (i <- 0 until config.numRuns) {
        val seed = System.currentTimeMillis()
        val r = new Random(seed)
        val paths = firstOrderWalk(init, nextFloat = r.nextFloat)
        save(paths, s"v${target.toString}-$i-s$seed")
      }
    }
  }

  def computeNumSteps(walkers: RDD[(Int, Array[Int])]) = {
    val bcWalkLength = context.broadcast(config.walkLength+1)
    walkers.map{case (_, path) => bcWalkLength.value - path.length}.reduce(_ + _)
  }

  def addAndRun(): Unit = {
    val g1 = readFromFile(config)
    val vertices: Array[Int] = g1.map(_._1).collect()
    val numSteps = Array.ofDim[Int](config.numRuns, vertices.length)
    config.rrType match {
      case RrType.m1 => {
        val init = initRandomWalk(g1)
        val ns = computeNumSteps(init)
        for (i <- 0 until config.numRuns) {
          numSteps(i) = Array.fill(vertices.length)(ns)
          val seed = System.currentTimeMillis()
          val r = new Random(seed)
          val paths = firstOrderWalk(init, nextFloat = r.nextFloat)
          save(paths, s"${config.rrType.toString}-$i-s$seed")
        }
      }
      case RrType.m2 => {
        val init = initRandomWalk(g1)
        for (i <- 0 until config.numRuns) {
          val seed = System.currentTimeMillis()
          val r = new Random(seed)
          val paths = firstOrderWalk(init, nextFloat = r.nextFloat)
          for (j <- 0 until vertices.length) {
            val target = vertices(j)
            logger.info(s"Added vertex $target")
            println(s"Added vertex $target")
            val afs1 = GraphMap.getNeighbors(target).map { case (v, w) => v }
            val fWalkers = filterUniqueWalkers(paths, afs1)
            val walkers = context.union(Array.fill(config.numWalks)(fWalkers))
            val ns = computeNumSteps(walkers)
            numSteps(i)(j)= ns
            val partialPaths = firstOrderWalk(walkers)
            val aws = fWalkers.map(tuple => tuple._1).collect()
            val unaffectedPaths = paths.filter { case p =>
              !aws.contains(p.head)
            }
            val newPaths = unaffectedPaths.union(partialPaths)
            save(newPaths, s"${config.rrType.toString}-v${target.toString}-$i-s$seed")
          }
        }
      }
      case RrType.m3 => {
        val init = initRandomWalk(g1)
        for (i <- 0 until config.numRuns) {
          val seed = System.currentTimeMillis()
          val r = new Random(seed)
          val paths = firstOrderWalk(init, nextFloat = r.nextFloat)
          for (j <- 0 until vertices.length) {
            val target = vertices(j)
            logger.info(s"Added vertex $target")
            println(s"Added vertex $target")
            val afs1 = GraphMap.getNeighbors(target).map { case (v, w) => v }
            val walkers = filterSplitPaths(paths, afs1)
            val ns = computeNumSteps(walkers)
            numSteps(i)(j)= ns
            val partialPaths = firstOrderWalk(walkers)
            val unaffectedPaths = filterNotAffectedPaths(paths, afs1)
            val newPaths = unaffectedPaths.union(partialPaths)
            save(newPaths, s"${config.rrType.toString}-v${target.toString}-$i-s$seed")
          }
        }
      }
      case RrType.m4 => {
        val init = initRandomWalk(g1)
        for (i <- 0 until config.numRuns) {
          val seed = System.currentTimeMillis()
          val r = new Random(seed)
          val paths = firstOrderWalk(init, nextFloat = r.nextFloat)
          for (j <- 0 until vertices.length) {
            val target = vertices(j)
            logger.info(s"Added vertex $target")
            println(s"Added vertex $target")
            val afs1 = GraphMap.getNeighbors(target).map { case (v, w) => v }
            val walkers = filterWalkers(paths, afs1)
            val ns = computeNumSteps(walkers)
            numSteps(i)(j)= ns
            val partialPaths = firstOrderWalk(walkers)
            val unaffectedPaths = filterNotAffectedPaths(paths, afs1)
            val newPaths = unaffectedPaths.union(partialPaths)
            save(newPaths, s"${config.rrType.toString}-v${target.toString}-$i-s$seed")
          }
        }

      }
    }
  }

  def filterUniqueWalkers(paths: RDD[Array[Int]], afs1: Array[Int]) = {
    filterWalkers(paths, afs1).reduceByKey((a, b) => a)
  }

  def filterWalkers(paths: RDD[Array[Int]], afs1: Array[Int]) = {
    filterAffectedPaths(paths, afs1).map(a => (a.head, Array(a.head)))
  }

  def filterSplitPaths(paths: RDD[Array[Int]], afs1: Array[Int]) = {
    filterAffectedPaths(paths, afs1).map { a =>
      val first = a.indexWhere(e => afs1.contains(e))
      (a.head, a.splitAt(first + 1)._1)
    }
  }

  def filterAffectedPaths(paths: RDD[Array[Int]], afs1: Array[Int]) = {
    paths.filter { case p =>
      !p.forall(s => !afs1.contains(s))
    }
  }

  def filterNotAffectedPaths(paths: RDD[Array[Int]], afs1: Array[Int]) = {
    paths.filter { case p =>
      p.forall(s => !afs1.contains(s))
    }
  }


  def removeVertex(g: RDD[(Int, Array[(Int, Float)])], target: Int): RDD[(Int, Array[(Int, Float)])] = {
    val bcTarget = context.broadcast(target)
    g.filter(_._1 != target).map { case (vId, neighbors) =>
      val filtered = neighbors.filter(_._1 != bcTarget.value)
      (vId, filtered)
    }
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


  def readFromFile(config: Params): RDD[(Int, Array[(Int, Float)])] = {
    // the directed and weighted parameters are only used for building the graph object.
    // is directed? they will be shared among stages and executors
    val bcDirected = context.broadcast(config.directed)
    val bcWeighted = context.broadcast(config.weighted) // is weighted?
    context.textFile(config.input, minPartitions
      = config
      .rddPartitions).flatMap { triplet =>
      val parts = triplet.split("\\s+")
      // if the weights are not specified it sets it to 1.0

      val weight = bcWeighted.value && parts.length > 2 match {
        case true => Try(parts.last.toFloat).getOrElse(1.0f)
        case false => 1.0f
      }

      val (src, dst) = (parts.head.toInt, parts(1).toInt)
      if (bcDirected.value) {
        Array((src, Array((dst, weight))), (dst, Array.empty[(Int, Float)]))
      } else {
        Array((src, Array((dst, weight))), (dst, Array((src, weight))))
      }
    }.
      reduceByKey(_ ++ _).
      partitionBy(partitioner).
      persist(StorageLevel.MEMORY_AND_DISK)
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
    if (pCount != config.numWalks * nVertices) {
      println(s"Inconsistent number of paths: nPaths=[${pCount}] != vertices[$nVertices]")
    }
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

  def save(paths: RDD[Array[Int]]): RDD[Array[Int]] = {

    paths.map {
      case (path) =>
        val pathString = path.mkString("\t")
        s"$pathString"
    }.repartition(config.rddPartitions).saveAsTextFile(s"${config.output}.${Property.pathSuffix}")
    paths
  }

  def save(paths: RDD[Array[Int]], suffix: String): RDD[Array[Int]] = {

    paths.map {
      case (path) =>
        val pathString = path.mkString("\t")
        s"$pathString"
    }.repartition(1).saveAsTextFile(s"${config.output}/${Property.removeAndRunSuffix}/$suffix")
    paths
  }

  def save(counts: Array[(Int, (Int, Int))]) = {

    context.parallelize(counts, config.rddPartitions).sortBy(_._2._2, ascending = false).map {
      case (vId, (count, occurs)) =>
        s"$vId\t$count\t$occurs"
    }.repartition(1).saveAsTextFile(s"${config.output}.${Property.countsSuffix}")
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
