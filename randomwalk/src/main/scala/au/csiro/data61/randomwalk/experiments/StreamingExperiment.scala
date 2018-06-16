package au.csiro.data61.randomwalk.experiments

import au.csiro.data61.randomwalk.algorithm.{GraphMap, UniformRandomWalk}
import au.csiro.data61.randomwalk.common.CommandParser.RrType
import au.csiro.data61.randomwalk.common._

import scala.collection.mutable
import scala.collection.parallel.ParSeq
import scala.util.control.Breaks._
import collection.JavaConverters._

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
      var totalTime: Long = 0
      GraphMap.reset
      WalkStorage.reset
      var prevWalks = ParSeq.empty[(Int, Int, Int, Seq[Int])]
      var deltaWalks = ParSeq.empty[(Int, Int, Seq[Int])]

      println("******* Building the graph ********")
      val (initEdges, edges) = config.grouped match {
        case false => fm.readPartitionEdgeListWithInitEdges(nr)
        case true => fm.readAlreadyPartitionedEdgeList()
      }

      val logSize = Math.min(edges.length, config.maxSteps) + 1
      if (numSteps.isEmpty) {
        numSteps = Array.ofDim[Int](config.numRuns, logSize)
        numWalkers = Array.ofDim[Int](config.numRuns, logSize)
        stepTimes = Array.ofDim[Long](config.numRuns, logSize)
        meanErrors = Array.ofDim[Double](config.numRuns, logSize)
        maxErrors = Array.ofDim[Double](config.numRuns, logSize)
      }

      // Construct initial graph
      println("******* Initialized Graph the graph ********")
      var afs = mutable.HashSet.empty[Int]
      if (!initEdges.isEmpty) {
        afs = updateGraph(initEdges)
        println(s"Number of edges: ${GraphMap.getNumEdges}")
        println(s"Number of vertices: ${GraphMap.getNumVertices}")
        val result = streamingAddAndRunWithId(afs, prevWalks)
        prevWalks = result._1
        //        deltaWalks = result._1._2
        totalTime = result._4
        numSteps(nr)(0) = result._2
        numWalkers(nr)(0) = result._3
        stepTimes(nr)(0) = result._4
        if (config.logErrors) {
          val (meanE, maxE): (Double, Double) = GraphUtils.computeErrorsMeanAndMax(result
            ._1, config)
          meanErrors(nr)(0) = meanE
          maxErrors(nr)(0) = maxE
          println(s"Mean Error: ${meanE}")
          println(s"Max Error: ${maxE}")
        }
        println(s"Total random walk time: $totalTime")
      } else {
        println("Initial graph is empty.")
      }
      println("Writing outputs...")
      fm.saveAffectedVertices(
        afs, s"${Property.afsSuffix}-${config.rrType.toString}-wl${
          config
            .walkLength
        }-nw${config.numWalks}-0-$nr")
      fm.savePaths(prevWalks, s"${config.rrType.toString}-wl${config.walkLength}-nw${
        config.numWalks
      }-0-$nr")
      fm.saveDegrees(GraphUtils.degrees(), s"${Property.degreeSuffix}-${
        config.rrType
          .toString
      }-wl${config.walkLength}-nw${config.numWalks}-0-$nr")

      breakable {
        println("Start streaming...")
        for (ec <- 1 until edges.length + 1) {
          val (step, updates) = edges(ec - 1)
          if (ec > config.maxSteps)
            break
          afs = updateGraph(updates)
          val result = streamingAddAndRunWithId(afs, prevWalks)
          prevWalks = result._1
          //          deltaWalks = result._1._2
          val ns = result._2
          val nw = result._3
          val stepTime = result._4
          totalTime += stepTime

          numSteps(nr)(ec) = ns
          numWalkers(nr)(ec) = nw
          stepTimes(nr)(ec) = stepTime

          val nEdges = GraphMap.getNumEdges
          println(s"Group-ID: ${step}")
          println(s"Step time: $stepTime")
          println(s"Total time: $totalTime")
          println(s"Number of edges: ${nEdges}")
          println(s"Number of vertices: ${GraphMap.getNumVertices}")
          println(s"Number of walks: ${prevWalks.size + deltaWalks.size}")
          if (config.logErrors) {
            val (meanE, maxE): (Double, Double) = GraphUtils.computeErrorsMeanAndMax(result._1,
              config)
            meanErrors(nr)(ec) = meanE
            maxErrors(nr)(ec) = maxE
            println(s"Mean Error: ${meanE}")
            println(s"Max Error: ${maxE}")
          }
          println(s"Number of actual steps: $ns")
          println(s"Number of actual walks: $nw")

          if (ec % config.savePeriod == 0) {
            //            if (config.rrType != RrType.m1) {
            fm.saveAffectedVertices(
              afs, s"${Property.afsSuffix}-${config.rrType.toString}-wl${
                config
                  .walkLength
              }-nw${config.numWalks}-$step-$nr")
            //            }
            fm.savePaths(prevWalks, s"${config.rrType.toString}-wl${
              config
                .walkLength
            }-nw${
              config.numWalks
            }-$step-$nr")
            //            fm.savePaths(deltaWalks, s"${config.rrType.toString}-delta-wl${config
            // .walkLength}-nw${
            //              config.numWalks
            //            }-$step-$nr")
            fm.saveDegrees(GraphUtils.degrees(), s"${Property.degreeSuffix}-${
              config.rrType
                .toString
            }-wl${config.walkLength}-nw${config.numWalks}-$step-$nr")
          }
        }
      }
      fm.savePaths(prevWalks, s"${config.rrType.toString}-wl${config.walkLength}-nw${
        config.numWalks
      }-final-$nr")
      //      fm.savePaths(deltaWalks, s"${config.rrType.toString}-delta-wl${config.walkLength}-nw${
      //        config.numWalks
      //      }-final-$nr")
      fm.saveDegrees(GraphUtils.degrees(), s"${Property.degreeSuffix}-${
        config.rrType
          .toString
      }-wl${config.walkLength}-nw${config.numWalks}-final-$nr")
      fm.saveAffectedVertices(
        afs, s"${Property.afsSuffix}-${config.rrType.toString}-wl${
          config
            .walkLength
        }-nw${config.numWalks}-final-$nr")
    }
    fm.saveComputations(numSteps, Property.stepsToCompute.toString)
    fm.saveComputations(numWalkers, Property.walkersToCompute.toString)
    fm.saveTimeSteps(stepTimes, Property.timesToCompute.toString)
    if (config.logErrors) {
      fm.saveErrors(meanErrors, Property.meanErrors.toString)
      fm.saveErrors(maxErrors, Property.maxErrors.toString)
    }
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

      GraphMap.putVertex(src, sNeighbors)
      afs.add(src)

      if (!config.directed) {
        dNeighbors ++= Seq((src, w))
        afs.add(dst)
        GraphMap.putVertex(dst, dNeighbors)
      }
    }

    return afs
  }

  def streamingAddAndRunWithId(afs: mutable.HashSet[Int], paths: ParSeq[(Int, Int, Int, Seq[Int])]):
  (ParSeq[(Int, Int, Int, Seq[Int])], Int, Int, Long) = {

    val result = config.rrType match {
      case RrType.m1 => {
        val sTime = System.currentTimeMillis()

        val init = rwalk.createWalkersByVertices(GraphMap.getVertices().par)
        val p = rwalk.secondOrderWalkWitIds(init)

        val tTime = System.currentTimeMillis() - sTime

        val ns = computeNumSteps(init)
        val nw = init.length

        (p.map { case (_, w) => (w._1, w._2, 1, w._3) }, ns, nw, tTime)
      }
      //      case RrType.m2 => {
      //        val sTime = System.currentTimeMillis()
      //
      //        var fWalkers: ParSeq[(Int, (Int, Int, Seq[Int]))] = filterUniqueWalkers(paths, afs)
      //        for (a <- afs) {
      //          if (fWalkers.count(_._1 == a) == 0) {
      //            fWalkers ++= ParSeq((a, (1, 0, Seq(a))))
      //          }
      //        }
      //        val walkers: ParSeq[(Int, (Int, Int, Seq[Int]))] = ParSeq.fill(config.numWalks)
      // (fWalkers)
      //          .flatten
      //        val pp = rwalk.secondOrderWalk(walkers)
      //
      //        val aws = fWalkers.map(tuple => tuple._1).seq
      //        val up = paths.filter { case p =>
      //          !aws.contains(p._3.head)
      //        }
      //        val np = up.union(pp)
      //
      //        val tTime = System.currentTimeMillis() - sTime
      //
      //        val ns = computeNumSteps(walkers)
      //        val nw = walkers.length
      //        ((np, pp), ns, nw, tTime)
      //      }
      case RrType.m3 => {
        val sTime = System.currentTimeMillis()

        val walkers = WalkStorage.filterAffectedPathsForM3(afs, config)
        val partialPaths = rwalk.secondOrderWalkWitIds(walkers)
        WalkStorage.updatePaths(partialPaths)

        val tTime = System.currentTimeMillis() - sTime

        val ns = computeNumStepsWithIds(walkers)
        val nw = walkers.length

        val newPathIds = partialPaths.map(_._1).toSet
        val allWalks = WalkStorage.getPathsWithIds().asScala.toSeq.par.map { case (wId, w) =>
          var isNew = 0
          if (newPathIds.contains(wId))
            isNew = 1
          (w._1, w._2, isNew, w._3)
        }
        (allWalks, ns, nw, tTime)
      }
      case RrType.m4 => {
        val sTime = System.currentTimeMillis()
        val walkers = WalkStorage.filterAffectedPathsForM4(afs, config)
        val partialPaths = rwalk.secondOrderWalkWitIds(walkers)
        WalkStorage.updatePaths(partialPaths)

        val tTime = System.currentTimeMillis() - sTime

        val ns = computeNumSteps(walkers)
        val nw = walkers.length
        val newPathIds = partialPaths.map(_._1).toSet
        val allWalks = WalkStorage.getPathsWithIds().asScala.toSeq.par.map { case (wId, w) =>
          var isNew = 0
          if (newPathIds.contains(wId))
            isNew = 1
          (w._1, w._2, isNew, w._3)
        }
        (allWalks, ns, nw, tTime)
        //        ((newPaths, partialPaths.map(a => a._2)), ns, nw, tTime)
      }
      case RrType.m5 => {
        val sTime = System.currentTimeMillis()

        var fWalkers: ParSeq[(Int, (Int, Int, Seq[Int]))] = paths.filter(a => afs.contains(a._4
          .head)).map(_._4.head).distinct.map(a => (a, (1, 0, Seq(a))))

        for (a <- afs) {
          if (fWalkers.count(_._1 == a) == 0) {
            fWalkers ++= ParSeq((a, (1, 0, Seq(a))))
          }
        }
        val walkers: ParSeq[(Int, (Int, Int, Seq[Int]))] = ParSeq.fill(config.numWalks)(fWalkers)
          .flatten
        val newWalks = rwalk.secondOrderWalkWitIds(walkers)

        //        val aws = fWalkers.map(tuple => tuple._1).seq
        val oldWalks = paths.filter { case p =>
          !afs.contains(p._4.head)
        }
        //        val np = up.union(pp)

        val tTime = System.currentTimeMillis() - sTime

        val allWalks = oldWalks.map { case w => (w._1, w._2, 0, w._4) }.union(newWalks.map { case
          (_, w) => (w._1, w._2, 1, w._3)
        })

        val ns = computeNumSteps(walkers)
        val nw = walkers.length
        (allWalks, ns, nw, tTime)
      }
    }

    result
  }

  def computeNumStepsWithIds(walkers: ParSeq[(Int, (Int, Int, Seq[Int]))]) = {
    println("%%%%% Computing number of steps %%%%%")
    val bcWalkLength = config.walkLength + 1
    walkers.map {
      case (_, path) => bcWalkLength - path._3.length
    }.reduce(_ + _)
  }

  def computeNumSteps(walkers: ParSeq[(Int, (Int, Int, Seq[Int]))]) = {
    println("%%%%% Computing number of steps %%%%%")
    val bcWalkLength = config.walkLength + 1
    walkers.map {
      case (_, path) => bcWalkLength - path._3.length
    }.reduce(_ + _)
  }

  def filterUniqueWalkers(paths: ParSeq[(Int, Int, Seq[Int])], afs1: mutable.Set[Int]) = {
    println("&&&&&&&&& filterUniqueWalkers &&&&&&&&&")
    filterWalkers(paths, afs1).groupBy(_._1).map {
      case (_, p) =>
        p.head
    }.toSeq
  }

  def filterUniqueWalkers(paths: ParSeq[(Int, Int, Seq[Int])], afs: mutable.HashSet[Int]) = {
    println("&&&&&&&&& filterUniqueWalkers &&&&&&&&&")
    filterWalkers(paths, afs).groupBy(_._1).map {
      case (_, p) =>
        p.head
    }.toSeq
  }

  def filterWalkers(paths: ParSeq[(Int, Int, Seq[Int])], afs1: mutable.Set[Int]): ParSeq[(Int, (Int,
    Int, Seq[Int]))] = {
    filterAffectedPaths(paths, afs1).map { case (wVersion, _, a) => (a.head, (wVersion, 0, Seq(a
      .head)))
    }
  }

  def filterWalkers(paths: ParSeq[(Int, Int, Seq[Int])], afs: mutable.HashSet[Int]): ParSeq[(Int,
    (Int, Int, Seq[Int]))] = {
    filterAffectedPaths(paths, afs).map { case (wVersion, _, a) => (a.head, (wVersion, 0, Seq(a
      .head)))
    }
  }

  def filterSplitPaths(paths: ParSeq[(Int, Int, Seq[Int])], afs1: mutable.Set[Int]) = {
    println("&&&&&&&&& filterSplitPaths &&&&&&&&&")
    filterAffectedPaths(paths, afs1).map {
      case (wVersion, _, a) =>
        val first = a.indexWhere(e => afs1.contains(e)) + 1
        (a.head, (wVersion, first, a.splitAt(first)._1))
    }
  }

  def filterSplitPaths(paths: ParSeq[(Int, Int, Seq[Int])], afs: mutable.HashSet[Int]) = {
    println("&&&&&&&&& filterSplitPaths &&&&&&&&&")
    filterAffectedPaths(paths, afs).map {
      case (wVersion, _, a) =>
        val first = a.indexWhere(e => afs.contains(e)) + 1
        (a.head, (wVersion, first, a.splitAt(first)._1))
    }
  }

  def filterAffectedPaths(paths: ParSeq[(Int, Int, Seq[Int])], afs1: mutable.Set[Int]) = {
    println("&&&&&&&&& filterAffectedPaths &&&&&&&&&")
    paths.filter {
      case (_, _, p) =>
        !p.forall(s => !afs1.contains(s))
    }
  }

  def filterAffectedPaths(paths: ParSeq[(Int, Int, Seq[Int])], afs: mutable.HashSet[Int]) = {
    println("&&&&&&&&& filterAffectedPaths &&&&&&&&&")
    paths.filter {
      case (_, _, p) =>
        !p.forall(s => !afs.contains(s))
    }
  }

  def filterUnaffectedPaths(paths: ParSeq[(Int, Int, Seq[Int])], afs1: mutable.Set[Int]) = {
    println("&&&&&&&&& filterUnaffectedPaths &&&&&&&&&")
    paths.filter {
      case (_, _, p) =>
        p.forall(s => !afs1.contains(s))
    }
  }

  def filterUnaffectedPaths(paths: ParSeq[(Int, Int, Seq[Int])], afs: mutable.HashSet[Int]) = {
    println("&&&&&&&&& filterUnaffectedPaths &&&&&&&&&")
    paths.filter {
      case (_, _, p) =>
        p.forall(s => !afs.contains(s))
    }
  }

}
