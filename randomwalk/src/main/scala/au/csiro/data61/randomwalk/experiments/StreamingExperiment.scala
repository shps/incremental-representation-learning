package au.csiro.data61.randomwalk.experiments

import au.csiro.data61.randomwalk.algorithm.{GraphMap, UniformRandomWalk}
import au.csiro.data61.randomwalk.common.CommandParser.RrType
import au.csiro.data61.randomwalk.common._

import scala.collection.mutable
import scala.collection.parallel.ParSeq

/**
  * Created by Hooman on 2018-04-11.
  */
case class StreamingExperiment(config: Params) {
  val fm = FileManager(config)
  val rwalk = UniformRandomWalk(config)

  def streamEdges(): Unit = {
    //        val edges = fm.readPartitionedEdgeList()
    var numSteps = Array.empty[Array[Int]]
    var numWalkers = Array.empty[Array[Int]]
    var stepTimes = Array.empty[Array[Long]]
    var meanErrors = Array.empty[Array[Double]]
    var maxErrors = Array.empty[Array[Double]]

    for (nr <- 0 until config.numRuns) {
      var totalTime:Long = 0
      GraphMap.reset
      WalkStorage.reset
      var prevWalks = ParSeq.empty[(Int, Seq[Int])]

      println("******* Building the graph ********")
      val (initEdges, edges) = fm.readPartitionedEdgeListWithInitEdges(nr)

      numSteps = Array.ofDim[Int](config.numRuns, edges.length)
      numWalkers = Array.ofDim[Int](config.numRuns, edges.length)
      meanErrors = Array.ofDim[Double](config.numRuns, edges.length)
      maxErrors = Array.ofDim[Double](config.numRuns, edges.length)

      // Construct initial graph
      println("******* Initialized Graph the graph ********")
      var afs = updateGraph(initEdges)
      println(s"Number of edges: ${GraphMap.getNumEdges}")
      println(s"Number of vertices: ${GraphMap.getNumVertices}")
      val result = streamingAddAndRunWithId(afs, prevWalks)
      prevWalks = result._1
      totalTime = result._4
      println(s"Total random walk time: $totalTime")

      var saveCount = 0
      for (ec <- 0 until edges.length) {
        saveCount += 1
        val (step, updates) = edges(ec)
        afs = updateGraph(updates)
        val result = streamingAddAndRunWithId(afs, prevWalks)
        prevWalks = result._1
        val ns = result._2
        val nw = result._3
        val stepTime = result._4
        totalTime += stepTime

        //        val (meanE, maxE): (Double, Double) = GraphUtils.computeErrorsMeanAndMax(result
        // ._1, config)
        numSteps(nr)(ec) = ns
        numWalkers(nr)(ec) = nw
        stepTimes(nr)(ec) = stepTime
        //        meanErrors(nr)(ec) = meanE
        //        maxErrors(nr)(ec) = maxE
        val nEdges = GraphMap.getNumEdges
        println(s"Step: ${step}")
        println(s"Step time: $stepTime")
        println(s"Total time: $totalTime")
        println(s"Number of edges: ${nEdges}")
        println(s"Number of vertices: ${GraphMap.getNumVertices}")
        println(s"Number of walks: ${prevWalks.size}")
        //        println(s"Mean Error: ${meanE}")
        //        println(s"Max Error: ${maxE}")
        println(s"Number of actual steps: $ns")
        println(s"Number of actual walks: $nw")

        if (saveCount % config.savePeriod == 0)
          fm.savePaths(prevWalks, s"${config.rrType.toString}-wl${config.walkLength}-nw${
            config.numWalks
          }-$step")
      }
      fm.savePaths(prevWalks, s"${config.rrType.toString}-wl${config.walkLength}-nw${
        config.numWalks
      }-final")
    }
    fm.saveComputations(numSteps, Property.stepsToCompute.toString)
    fm.saveComputations(numWalkers, Property.walkersToCompute.toString)
    fm.saveErrors(meanErrors, Property.meanErrors.toString)
    fm.saveErrors(maxErrors, Property.maxErrors.toString)
  }

  def updateGraph(updates: Seq[(Int, Int)]): mutable.HashSet[Int] = {
    val afs = new mutable.HashSet[Int]()
    for (u <- updates) {
      val src = u._1
      val dst = u._2
      val w = 1f
      var sNeighbors = GraphMap.getNeighbors(src)
      var dNeighbors = GraphMap.getNeighbors(dst)
      sNeighbors ++= Seq((dst, w))
      dNeighbors ++= Seq((src, w))
      GraphMap.putVertex(src, sNeighbors)
      GraphMap.putVertex(dst, dNeighbors)

      afs.add(src)
      afs.add(dst)
    }

    return afs
  }

  def streamingAddAndRunWithId(afs: mutable.HashSet[Int], paths: ParSeq[(Int, Seq[Int])]):
  (ParSeq[(Int, Seq[Int])], Int, Int, Long) = {

    val result = config.rrType match {
      case RrType.m1 => {
        val sTime = System.currentTimeMillis()

        val init = rwalk.createWalkersByVertices(GraphMap.getVertices().par)
        val p = rwalk.secondOrderWalk(init)

        val tTime = System.currentTimeMillis() - sTime

        val ns = computeNumSteps(init)
        val nw = init.length

        (p, ns, nw, tTime)
      }
      case RrType.m2 => {
        val sTime = System.currentTimeMillis()

        var fWalkers: ParSeq[(Int, (Int, Seq[Int]))] = filterUniqueWalkers(paths, afs)
        for (a <- afs) {
          if (fWalkers.count(_._1 == a) == 0) {
            fWalkers ++= ParSeq((a, (1, Seq(a))))
          }
        }
        val walkers: ParSeq[(Int, (Int, Seq[Int]))] = ParSeq.fill(config.numWalks)(fWalkers).flatten
        val pp = rwalk.secondOrderWalk(walkers)

        val aws = fWalkers.map(tuple => tuple._1).seq
        val up = paths.filter { case p =>
          !aws.contains(p._2.head)
        }
        val np = up.union(pp)

        val tTime = System.currentTimeMillis() - sTime

        val ns = computeNumSteps(walkers)
        val nw = walkers.length
        (np, ns, nw, tTime)
      }
      case RrType.m3 => {
        val sTime = System.currentTimeMillis()

        val walkers = WalkStorage.filterAffectedPathsForM3(afs, config)
        val partialPaths = rwalk.secondOrderWalkWitIds(walkers)
        WalkStorage.updatePaths(partialPaths)
        val newPaths = WalkStorage.getPaths()

        val tTime = System.currentTimeMillis() - sTime

        val ns = computeNumStepsWithIds(walkers)
        val nw = walkers.length
        (newPaths, ns, nw, tTime)
      }
      case RrType.m4 => {
        val sTime = System.currentTimeMillis()
        val walkers = WalkStorage.filterAffectedPathsForM4(afs, config)
//        var walkers = filterWalkers(paths, afs)
//        for (a <- afs) {
//          if (walkers.count(_._1 == a) == 0) {
//            walkers ++= rwalk.initWalker(a)
//          }
//        }
        val partialPaths = rwalk.secondOrderWalkWitIds(walkers)
//        val unaffectedPaths = filterUnaffectedPaths(paths, afs)
//        val newPaths = unaffectedPaths.union(partialPaths)
        WalkStorage.updatePaths(partialPaths)
        val newPaths = WalkStorage.getPaths()

        val tTime = System.currentTimeMillis() - sTime

        val ns = computeNumSteps(walkers)
        val nw = walkers.length
        (newPaths, ns, nw, tTime)
      }
    }

    result
  }

  def computeNumStepsWithIds(walkers: ParSeq[(Int, (Int, Seq[Int]))]) = {
    println("%%%%% Computing number of steps %%%%%")
    val bcWalkLength = config.walkLength + 1
    walkers.map {
      case (_, path) => bcWalkLength - path._2.length
    }.reduce(_ + _)
  }

  def computeNumSteps(walkers: ParSeq[(Int, (Int, Seq[Int]))]) = {
    println("%%%%% Computing number of steps %%%%%")
    val bcWalkLength = config.walkLength + 1
    walkers.map {
      case (_, path) => bcWalkLength - path._2.length
    }.reduce(_ + _)
  }

  def filterUniqueWalkers(paths: ParSeq[(Int, Seq[Int])], afs1: mutable.Set[Int]) = {
    println("&&&&&&&&& filterUniqueWalkers &&&&&&&&&")
    filterWalkers(paths, afs1).groupBy(_._1).map {
      case (_, p) =>
        p.head
    }.toSeq
  }

  def filterUniqueWalkers(paths: ParSeq[(Int, Seq[Int])], afs: mutable.HashSet[Int]) = {
    println("&&&&&&&&& filterUniqueWalkers &&&&&&&&&")
    filterWalkers(paths, afs).groupBy(_._1).map {
      case (_, p) =>
        p.head
    }.toSeq
  }

  def filterWalkers(paths: ParSeq[(Int, Seq[Int])], afs1: mutable.Set[Int]): ParSeq[(Int, (Int,
    Seq[Int]))] = {
    filterAffectedPaths(paths, afs1).map { case (wVersion, a) => (a.head, (wVersion, Seq(a.head))) }
  }

  def filterWalkers(paths: ParSeq[(Int, Seq[Int])], afs: mutable.HashSet[Int]): ParSeq[(Int,
    (Int, Seq[Int]))] = {
    filterAffectedPaths(paths, afs).map { case (wVersion, a) => (a.head, (wVersion, Seq(a.head))) }
  }

  def filterSplitPaths(paths: ParSeq[(Int, Seq[Int])], afs1: mutable.Set[Int]) = {
    println("&&&&&&&&& filterSplitPaths &&&&&&&&&")
    filterAffectedPaths(paths, afs1).map {
      case (wVersion, a) =>
        val first = a.indexWhere(e => afs1.contains(e))
        (a.head, (wVersion, a.splitAt(first + 1)._1))
    }
  }

  def filterSplitPaths(paths: ParSeq[(Int, Seq[Int])], afs: mutable.HashSet[Int]) = {
    println("&&&&&&&&& filterSplitPaths &&&&&&&&&")
    filterAffectedPaths(paths, afs).map {
      case (wVersion, a) =>
        val first = a.indexWhere(e => afs.contains(e))
        (a.head, (wVersion, a.splitAt(first + 1)._1))
    }
  }

  def filterAffectedPaths(paths: ParSeq[(Int, Seq[Int])], afs1: mutable.Set[Int]) = {
    println("&&&&&&&&& filterAffectedPaths &&&&&&&&&")
    paths.filter {
      case (_, p) =>
        !p.forall(s => !afs1.contains(s))
    }
  }

  def filterAffectedPaths(paths: ParSeq[(Int, Seq[Int])], afs: mutable.HashSet[Int]) = {
    println("&&&&&&&&& filterAffectedPaths &&&&&&&&&")
    paths.filter {
      case (_, p) =>
        !p.forall(s => !afs.contains(s))
    }
  }

  def filterUnaffectedPaths(paths: ParSeq[(Int, Seq[Int])], afs1: mutable.Set[Int]) = {
    println("&&&&&&&&& filterUnaffectedPaths &&&&&&&&&")
    paths.filter {
      case (_, p) =>
        p.forall(s => !afs1.contains(s))
    }
  }

  def filterUnaffectedPaths(paths: ParSeq[(Int, Seq[Int])], afs: mutable.HashSet[Int]) = {
    println("&&&&&&&&& filterUnaffectedPaths &&&&&&&&&")
    paths.filter {
      case (_, p) =>
        p.forall(s => !afs.contains(s))
    }
  }

}
