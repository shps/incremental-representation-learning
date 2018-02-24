package au.csiro.data61.randomwalk.algorithm

import au.csiro.data61.randomwalk.common.CommandParser.RrType
import au.csiro.data61.randomwalk.common.{FileManager, Params, Property}

import scala.util.Random

/**
  * Created by Hooman on 2018-02-16.
  */
case class Experiments(config: Params) extends Serializable {

  val fm = FileManager(config)
  val rwalk = UniformRandomWalk(config)
  val ADD = 1
  val REM = 0

  def addAndRun(): Unit = {
    val g1 = fm.readFromFile()
    val vertices: Array[Int] = g1.map(_._1).sortWith((a, b) => a < b)
    val numSteps = Array.ofDim[Int](config.numRuns, vertices.length)
    val numWalkers = Array.ofDim[Int](config.numRuns, vertices.length)
    for (i <- 0 until config.numRuns) {
      for (j <- 0 until math.min(config.numVertices, vertices.length)) {
        val target = vertices(j)
        val g2 = removeVertex(g1, target)
        val init = rwalk.initRandomWalk(g2)
        val paths = rwalk.firstOrderWalk(init)
        rwalk.buildGraphMap(g1)
        println(s"Added vertex $target")
        val afs1 = GraphMap.getNeighbors(target).map { case (v, _) => v }
        val newPaths = config.rrType match {
          case RrType.m1 => {
            val init = rwalk.initRandomWalk(g1)
            val ns = computeNumSteps(init)
            val nw = computeNumWalkers(init)
            numSteps(i) = Array.fill(vertices.length)(ns)
            numWalkers(i) = Array.fill(vertices.length)(nw)
            val p = rwalk.firstOrderWalk(init)
            p
          }
          case RrType.m2 => {
            val fWalkers: Array[(Int, Array[Int])] = filterUniqueWalkers(
              paths, afs1) ++ Array((target, Array(target)))
            val walkers: Array[(Int, Array[Int])] = Array.fill(config
              .numWalks)(fWalkers).flatMap(a => a)
            val ns = computeNumSteps(walkers)
            val nw = computeNumWalkers(walkers)
            numSteps(i)(j) = ns
            numWalkers(i)(j) = nw
            val pp = rwalk.firstOrderWalk(walkers)
            val aws = fWalkers.map(tuple => tuple._1)
            val up = paths.filter { case p =>
              !aws.contains(p.head)
            }

            val np = up.union(pp)
            np
          }
          case RrType.m3 => {
            val walkers = filterSplitPaths(paths, afs1).union(rwalk.initWalker(target))
            val ns = computeNumSteps(walkers)
            val nw = computeNumWalkers(walkers)
            numSteps(i)(j) = ns
            numWalkers(i)(j) = nw
            val partialPaths = rwalk.firstOrderWalk(walkers)
            val unaffectedPaths = filterUnaffectedPaths(paths, afs1)
            val newPaths = unaffectedPaths.union(partialPaths)
            newPaths
          }
          case RrType.m4 => {
            val walkers = filterWalkers(paths, afs1).union(rwalk.initWalker(target))
            val ns = computeNumSteps(walkers)
            val nw = computeNumWalkers(walkers)
            numSteps(i)(j) = ns
            numWalkers(i)(j) = nw
            val partialPaths = rwalk.firstOrderWalk(walkers)
            val unaffectedPaths = filterUnaffectedPaths(paths, afs1)
            val newPaths = unaffectedPaths.union(partialPaths)
            newPaths
          }
        }

        fm.save(newPaths, s"${config.rrType.toString}-wl${config.walkLength}-nw${
          config
            .numWalks
        }-v${target.toString}-$i")
      }

    }


    fm.save(vertices, numSteps, Property.stepsToCompute.toString)
    fm.save(vertices, numWalkers, Property.walkersToCompute.toString)
  }

  def extractEdges(g1: Array[(Int, Array[(Int, Float)])]): Array[(Int, Int)] = {
    g1.flatMap { case (src, neighbors) =>
      neighbors.map { case (dst, _) => (src, dst) }
    }
  }

  def streamingUpdates(): Unit = {
    val g1 = fm.readFromFile()
    val vertices: Array[Int] = g1.map(_._1).sortWith((a, b) => a < b)
    val edges: Array[(Int, Int)] = extractEdges(g1)
    print(s"Number of edges: ${edges.length}")
    var g2 = null
    for (i <- 0 until config.numRuns) {
      val rand = new Random(config.seed + i)
//      val sEdges = rand.shuffle(edges)
//      for (e <- sEdges) {
//
//      }

    }
  }


  def removeAndRun(): Unit = {
    val g1 = fm.readFromFile()
    val vertices: Array[Int] = g1.map(_._1)
    for (target <- vertices) {
      println(s"Removed vertex $target")
      val g2 = removeVertex(g1, target)
      val init = rwalk.initRandomWalk(g2)
      for (i <- 0 until config.numRuns) {
        val seed = System.currentTimeMillis()
        val r = new Random(seed)
        val paths = rwalk.firstOrderWalk(init, nextFloat = r.nextFloat)
        fm.save(paths, s"v${
          target.toString
        }-$i-s$seed")
      }
    }
  }

  def removeVertex(g: Array[(Int, Array[(Int, Float)])], target: Int): Array[(Int, Array[(Int,
    Float)])] = {
    g.filter(_._1 != target).map {
      case (vId, neighbors) =>
        val filtered = neighbors.filter(_._1 != target)
        (vId, filtered)
    }
  }

  def filterUniqueWalkers(paths: Array[Array[Int]], afs1: Array[Int]) = {
    filterWalkers(paths, afs1).groupBy(_._1).map { case (_, p) =>
      p.head
    }.toArray
  }

  def filterWalkers(paths: Array[Array[Int]], afs1: Array[Int]): Array[(Int, Array[Int])] = {
    filterAffectedPaths(paths, afs1).map(a => (a.head, Array(a.head)))
  }

  def filterSplitPaths(paths: Array[Array[Int]], afs1: Array[Int]) = {
    filterAffectedPaths(paths, afs1).map {
      a =>
        val first = a.indexWhere(e => afs1.contains(e))
        (a.head, a.splitAt(first + 1)._1)
    }
  }

  def filterAffectedPaths(paths: Array[Array[Int]], afs1: Array[Int]) = {
    paths.filter {
      case p =>
        !p.forall(s => !afs1.contains(s))
    }
  }

  def filterUnaffectedPaths(paths: Array[Array[Int]], afs1: Array[Int]) = {
    paths.filter {
      case p =>
        p.forall(s => !afs1.contains(s))
    }
  }

  def computeNumSteps(walkers: Array[(Int, Array[Int])]) = {
    val bcWalkLength = config.walkLength + 1
    walkers.map {
      case (_, path) => bcWalkLength - path.length
    }.reduce(_ + _)
  }

  def computeNumWalkers(walkers: Array[(Int, Array[Int])]) = {
    walkers.length
  }
}
