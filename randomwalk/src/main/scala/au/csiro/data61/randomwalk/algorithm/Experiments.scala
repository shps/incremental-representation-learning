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
      for (j <- 0 until vertices.length) {
        val target = vertices(j)
        val g2 = removeVertex(g1, target)
        val init = rwalk.initRandomWalk(g2)
        val paths = rwalk.firstOrderWalk(init)
        //            println(s"Before adding $target")
        //            checkGraphMap()
        rwalk.buildGraphMap(g1)
        println(s"Added vertex $target")
        val afs1 = GraphMap.getNeighbors(target).map { case (v, _) => v }
        config.rrType match {
          case RrType.m1 => {
            val init = rwalk.initRandomWalk(g1)
            //        checkGraphMap()
            val ns = computeNumSteps(init)
            numSteps(i) = Array.fill(vertices.length)(ns)
            val p = rwalk.firstOrderWalk(init)
            //          val summary = p.map(a => (a.head, (1, a.length))).reduceByKey((a,b)=>(a
            // ._1+b._1,a._2+b._2))
            //          println(s"Partial Paths: ${summary.count()}\n${summary.collect()
            // .mkString("
            // ")}")
            fm.save(p, s"${config.rrType.toString}-$i")
          }
          case RrType.m2 => {
            //            println(s"+$target -> ${afs1.mkString(" ")}")
            //            checkGraphMap()
            val fWalkers: Array[(Int, Array[Int])] = filterUniqueWalkers(
              paths, afs1).collect() ++ Array((target, Array(target)))
            //            println(s"Unique Walkers: ${fWalkers.length}\tdistinct: ${fWalkers
            // .map(_._1).distinct.length}")
            val walkers: RDD[(Int, Array[Int])] = context.parallelize(Array.fill(config
              .numWalks)(fWalkers).flatMap(a => a), numSlices = config.rddPartitions)
            //            println(s"Walkers: ${walkers.count()}\tdistinct: ${walkers.map(_._1)
            // .distinct().count()}\nEach Key: ${walkers.map { case (a, _) => (a, 1) }
            // .reduceByKey(_ + _).collect().mkString(" ")}")
            val ns = computeNumSteps(walkers)
            numSteps(i)(j) = ns
            val pp = rwalk.firstOrderWalk(walkers)
            //            val summary = pp.map(a => (a.head, (1, a.length))).reduceByKey((a,b)
            // =>(a._1+b._1,a._2+b._2))
            //            println(s"Partial Paths: ${summary.count()}\n${summary.collect()
            // .mkString(" ")}")
            val aws = fWalkers.map(tuple => tuple._1)
            val up = paths.filter { case p =>
              !aws.contains(p.head)
            }

            //            println(s"Unaffected paths: ${up.count()}\t${up.map(a => (a.head, 1))
            // .reduceByKey(_ + _).collect().mkString(" ")}")
            val np = up.union(pp)
            //            val summary2 = np.map(a => (a.head, (1, a.length))).reduceByKey((a,b)
            // =>(a._1+b._1,a._2+b._2))
            //            println(s"Partial Paths: ${summary2.count()}\n${summary2.collect()
            // .mkString(" ")}")
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
    //    filterAffectedPaths(paths, afs1).map { a =>
    //      val first = a.indexWhere(e => afs1.contains(e))
    //      (a.head, a.splitAt(first + 1)._1)
    //    }
    filterAffectedPaths(paths, afs1).map {
      a =>
        (a.head, Array(a.head))
    }
  }

  def filterAffectedPaths(paths: RDD[Array[Int]], afs1: Array[Int]) = {
    //    paths.filter { case p =>
    //      !p.forall(s => !afs1.contains(s))
    //    }
    paths
  }

  def filterUnaffectedPaths(paths: RDD[Array[Int]], afs1: Array[Int]) = {
    //    paths.filter { case p =>
    //      p.forall(s => !afs1.contains(s))
    //    }
    paths.filter {
      case p =>
        false
    }
  }

  def computeNumSteps(walkers: RDD[(Int, Array[Int])]) = {
    val bcWalkLength = context.broadcast(config.walkLength + 1)
    walkers.map {
      case (_, path) => bcWalkLength.value - path.length
    }.reduce(_ + _)
  }

}
