package au.csiro.data61.randomwalk.algorithm

import au.csiro.data61.randomwalk.common.CommandParser.{RrType, WalkType}
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
        val edges = fm.readPartitionedEdgeList()
//    val (initEdges, edges) = fm.readPartitionedEdgeListWithInitEdges()
    val numSteps = Array.ofDim[Int](config.numRuns, edges.length)
    val numWalkers = Array.ofDim[Int](config.numRuns, edges.length)
    val meanErrors = Array.ofDim[Double](config.numRuns, edges.length)
    val maxErrors = Array.ofDim[Double](config.numRuns, edges.length)


    for (nr <- 0 until config.numRuns) {
      GraphMap.reset
      WalkStorage.reset
      var prevWalks = ParSeq.empty[(Int, Seq[Int])]

//      println("******* Building the graph ********")
//      for (u <- initEdges) {
//        val src = u._1
//        val dst = u._2
//        val w = 1f
//        var sNeighbors = GraphMap.getNeighbors(src)
//        var dNeighbors = GraphMap.getNeighbors(dst)
//        sNeighbors ++= Seq((dst, w))
//        dNeighbors ++= Seq((src, w))
//        GraphMap.putVertex(src, sNeighbors)
//        GraphMap.putVertex(dst, dNeighbors)
//      }
//
//      println(s"Number of edges: ${GraphMap.getNumEdges}")
//      println(s"Number of vertices: ${GraphMap.getNumVertices}")
//
//      println("******* Initialized Graph the graph ********")
//      val result = streamingAddAndRunWithId(initEdges, prevWalks)
//      prevWalks = result._1

      var saveCount = 0
      for (ec <- 0 until edges.length) {
        saveCount += 1
        val (year, updates) = edges(ec)
        val result = streamingAddAndRunWithId(updates, prevWalks)
        prevWalks = result._1
        val ns = result._2
        val nw = result._3

        //        val (meanE, maxE): (Double, Double) = GraphUtils.computeErrorsMeanAndMax(result
        // ._1, config)
        numSteps(nr)(ec) = ns
        numWalkers(nr)(ec) = nw
        //        meanErrors(nr)(ec) = meanE
        //        maxErrors(nr)(ec) = maxE
        val nEdges = GraphMap.getNumEdges
        println(s"Step: ${year}")
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
          }-$year")
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

  def streamingAddAndRunWithId(updates: Seq[(Int, Int)], paths: ParSeq[(Int, Seq[Int])]):
  (ParSeq[(Int, Seq[Int])],
    Int, Int) = {
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

      //      println(s"Added Edge: $src <-> $dst")
      afs.add(src)
      afs.add(dst)
    }

    println("******* Updated the graph ********")


    val result = config.rrType match {
      case RrType.m1 => {
        val init = rwalk.createWalkersByVertices(GraphMap.getVertices().par)
        //        val ns = 0
        //        val nw = 0
        val ns = computeNumSteps(init)
        val nw = init.length
        val p = rwalk.secondOrderWalk(init)
        (p, ns, nw)
      }
      case RrType.m2 => {
        var fWalkers: ParSeq[(Int, (Int, Seq[Int]))] = filterUniqueWalkers(paths, afs)
        for (a <- afs) {
          if (fWalkers.count(_._1 == a) == 0) {
            fWalkers ++= ParSeq((a, (1, Seq(a))))
          }
        }
        val walkers: ParSeq[(Int, (Int, Seq[Int]))] = ParSeq.fill(config.numWalks)(fWalkers).flatten
        val ns = computeNumSteps(walkers)
        val nw = walkers.length

        val pp = rwalk.secondOrderWalk(walkers)

        val aws = fWalkers.map(tuple => tuple._1).seq
        val up = paths.filter { case p =>
          !aws.contains(p._2.head)
        }

        val np = up.union(pp)
        (np, ns, nw)
      }
      case RrType.m3 => {
        val walkers = WalkStorage.filterAffectedPaths(afs, config)
        //        val ns = 0
        //        val nw = 0
        val ns = computeNumStepsWithIds(walkers)
        val nw = walkers.length
        val partialPaths = rwalk.secondOrderWalkWitIds(walkers)
        WalkStorage.updatePaths(partialPaths)
        //        val unaffectedPaths = filterUnaffectedPaths(paths, afs)
        //        val newPaths = unaffectedPaths.union(partialPaths)
        val newPaths = WalkStorage.getPaths()
        (newPaths, ns, nw)
      }
      case RrType.m4 => {
        var walkers = filterWalkers(paths, afs)
        for (a <- afs) {
          if (walkers.count(_._1 == a) == 0) {
            walkers ++= rwalk.initWalker(a)
          }
        }

        val ns = computeNumSteps(walkers)
        val nw = walkers.length
        val partialPaths = rwalk.secondOrderWalk(walkers)
        val unaffectedPaths = filterUnaffectedPaths(paths, afs)
        val newPaths = unaffectedPaths.union(partialPaths)
        (newPaths, ns, nw)
      }
    }

    result
  }

  def computeNumStepsWithIds(walkers: ParSeq[(Int, (Int, Seq[Int]))]) = {
    println("%%%%% Compuing number of steps %%%%%")
    val bcWalkLength = config.walkLength + 1
    walkers.map {
      case (_, path) => bcWalkLength - path._2.length
    }.reduce(_ + _)
  }

  def computeNumSteps(walkers: ParSeq[(Int, (Int, Seq[Int]))]) = {
    println("%%%%% Compuing number of steps %%%%%")
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
