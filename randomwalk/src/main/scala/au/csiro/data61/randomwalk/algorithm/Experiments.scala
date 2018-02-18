package au.csiro.data61.randomwalk.algorithm

import au.csiro.data61.randomwalk.common.CommandParser.RrType
import au.csiro.data61.randomwalk.common.{FileManager, Params}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
  * Created by Hooman on 2018-02-16.
  */
case class Experiments(context: SparkContext, config: Params) extends Serializable {

  val fm = FileManager(context, config)
  val rwalk = UniformRandomWalk(context, config)

  def addAndRun(): Unit = {
    val g1 = fm.readFromFile()
    val vertices: Array[Int] = g1.map(_._1).collect()
    val numSteps = Array.ofDim[Int](config.numRuns, vertices.length)
    for (i <- 0 until config.numRuns) {
      config.rrType match {
        case RrType.m1 => {
          val init = rwalk.initRandomWalk(g1)
          val ns = computeNumSteps(init)
          numSteps(i) = Array.fill(vertices.length)(ns)
          val p = rwalk.firstOrderWalk(init)
          fm.save(p, s"${config.rrType.toString}-$i")
        }
        case _ =>
          for (j <- 0 until vertices.length) {
            val target = vertices(j)
            val g2 = removeVertex(g1, target)
            val init = rwalk.initRandomWalk(g2)
            val paths = rwalk.firstOrderWalk(init)
            rwalk.buildGraphMap(g1)
            println(s"Added vertex $target")
            val afs1 = GraphMap.getNeighbors(target).map { case (v, _) => v }
            config.rrType match {
              case RrType.m2 => {
                val fWalkers: Array[(Int, Array[Int])] = filterUniqueWalkers(
                  paths, afs1).collect() ++ Array((target, Array(target)))
                val walkers: RDD[(Int, Array[Int])] = context.parallelize(Array.fill(config
                  .numWalks)(fWalkers).flatMap(a => a), numSlices = config.rddPartitions)
                val ns = computeNumSteps(walkers)
                numSteps(i)(j) = ns
                val pp = rwalk.firstOrderWalk(walkers)
                val aws = fWalkers.map(tuple => tuple._1)
                val up = paths.filter { case p =>
                  !aws.contains(p.head)
                }

                val np = up.union(pp)
                fm.save(np, s"${config.rrType.toString}-v${target.toString}-$i")
              }
              case RrType.m3 => {
                val walkers = filterSplitPaths(paths, afs1).union(rwalk.initWalker(target))
                val ns = computeNumSteps(walkers)
                numSteps(i)(j) = ns
                val partialPaths = rwalk.firstOrderWalk(walkers)
                val unaffectedPaths = filterUnaffectedPaths(paths, afs1)
                val newPaths = unaffectedPaths.union(partialPaths)
                fm.save(newPaths, s"${config.rrType.toString}-v${target.toString}-$i")
              }
              case RrType.m4 => {
                val walkers = filterWalkers(paths, afs1).union(rwalk.initWalker(target))
                val ns = computeNumSteps(walkers)
                numSteps(i)(j) = ns
                val partialPaths = rwalk.firstOrderWalk(walkers)
                val unaffectedPaths = filterUnaffectedPaths(paths, afs1)
                val newPaths = unaffectedPaths.union(partialPaths)
                fm.save(newPaths, s"${config.rrType.toString}-v${target.toString}-$i")
              }

            }
          }

      }
    }

    fm.save(vertices, numSteps)
  }


  def removeAndRun(): Unit = {
    val g1 = fm.readFromFile()
    val vertices: Array[Int] = g1.map(_._1).collect()
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

  def removeVertex(g: RDD[(Int, Array[(Int, Float)])], target: Int): RDD[(Int, Array[(Int, Float)
    ])] = {
    val bcTarget = context.broadcast(target)
    g.filter(_._1 != target).map {
      case (vId, neighbors) =>
        val filtered = neighbors.filter(_._1 != bcTarget.value)
        (vId, filtered)
    }
  }

  def filterUniqueWalkers(paths: RDD[Array[Int]], afs1: Array[Int]) = {
    filterWalkers(paths, afs1).reduceByKey((a, b) => a)
  }

  def filterWalkers(paths: RDD[Array[Int]], afs1: Array[Int]): RDD[(Int, Array[Int])] = {
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

  def filterUnaffectedPaths(paths: RDD[Array[Int]], afs1: Array[Int]) = {
    paths.filter { case p =>
      p.forall(s => !afs1.contains(s))
    }
  }

  def computeNumSteps(walkers: RDD[(Int, Array[Int])]) = {
    val bcWalkLength = context.broadcast(config.walkLength + 1)
    walkers.map {
      case (_, path) => bcWalkLength.value - path.length
    }.reduce(_ + _)
  }

}
