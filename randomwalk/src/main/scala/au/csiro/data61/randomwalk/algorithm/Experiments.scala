package au.csiro.data61.randomwalk.algorithm

import au.csiro.data61.randomwalk.common.CommandParser.{RrType, WalkType}
import au.csiro.data61.randomwalk.common._

import scala.collection.mutable
import scala.collection.parallel.ParSeq
import scala.util.Random

/**
  * Created by Hooman on 2018-02-16.
  */
case class Experiments(config: Params) {


  val fm = FileManager(config)
  val rwalk = UniformRandomWalk(config)
  val ADD = 1
  val REM = 0

  def addAndRun(): Unit = {
    val g1 = fm.readFromFile(config.directed)
    val vertices: Seq[Int] = g1.map(_._1).seq.sortWith((a, b) => a < b)
    val numSteps = Array.ofDim[Int](config.numRuns, vertices.length)
    val numWalkers = Array.ofDim[Int](config.numRuns, vertices.length)

    for (i <- 0 until config.numRuns) {
      for (j <- 0 until math.min(config.numVertices, vertices.length)) {
        val target = vertices(j)
        val g2 = removeVertex(g1.par, target)
        val paths = config.wType match {
          case WalkType.firstorder => rwalk.firstOrderWalk(rwalk.initRandomWalk(g2))
          case WalkType.secondorder => rwalk.secondOrderWalk(rwalk.initRandomWalk(g2))
        }
        rwalk.buildGraphMap(g1.seq)
        println(s"Added vertex $target")
        val afs1 = GraphMap.getNeighbors(target).map { case (v, _) => v }
        val newPaths = config.rrType match {
          case RrType.m1 => {
            val walkers = rwalk.initRandomWalk(g1)
            val ns = computeNumSteps(walkers)
            val nw = computeNumWalkers(walkers)
            numSteps(i) = Array.fill(vertices.length)(ns)
            numWalkers(i) = Array.fill(vertices.length)(nw)
            val p = config.wType match {
              case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
              case WalkType.secondorder => rwalk.secondOrderWalk(walkers)
            }
            p
          }
          case RrType.m2 => {
            val fWalkers: ParSeq[(Int, (Int, Seq[Int]))] = filterUniqueWalkers(
              paths, afs1) ++ ParSeq((target, (1, Seq(target))))
            val walkers: ParSeq[(Int, (Int, Seq[Int]))] = ParSeq.fill(config
              .numWalks)(fWalkers).flatten
            val ns = computeNumSteps(walkers)
            val nw = computeNumWalkers(walkers)
            numSteps(i)(j) = ns
            numWalkers(i)(j) = nw
            val pp = config.wType match {
              case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
              case WalkType.secondorder => rwalk.secondOrderWalk(walkers)
            }
            val aws = fWalkers.map(tuple => tuple._1).seq
            val up = paths.filter { case (_, p) =>
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
            val partialPaths = config.wType match {
              case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
              case WalkType.secondorder => rwalk.secondOrderWalk(walkers)
            }
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
            val partialPaths = config.wType match {
              case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
              case WalkType.secondorder => rwalk.secondOrderWalk(walkers)
            }
            val unaffectedPaths = filterUnaffectedPaths(paths, afs1)
            val newPaths = unaffectedPaths.union(partialPaths)
            newPaths
          }
        }

        fm.savePaths(newPaths, s"${config.rrType.toString}-wl${config.walkLength}-nw${
          config
            .numWalks
        }-v${target.toString}-$i")
      }

    }


    fm.saveNumSteps(vertices, numSteps, Property.stepsToCompute.toString)
    fm.saveNumSteps(vertices, numWalkers, Property.walkersToCompute.toString)
  }

  def extractEdges(g1: ParSeq[(Int, mutable.Set[(Int, Float)])]): ParSeq[(Int, (Int, Float))] = {
    g1.flatMap { case (src, neighbors) =>
      neighbors.map { case (dst, w) => (src, (dst, w)) }
    }
  }

  def streamingUpdates(): Unit = {
    val g1 = fm.readFromFile(directed = true) // read it as directed
    val edges: ParSeq[(Int, (Int, Float))] = extractEdges(g1)
    print(s"Number of edges: ${edges.length}")
    val rand = new Random(config.seed)
    val sEdges = rand.shuffle(edges.seq)
    val numSteps = Array.ofDim[Int](config.numRuns, sEdges.length)
    val numWalkers = Array.ofDim[Int](config.numRuns, sEdges.length)
    val meanErrors = Array.ofDim[Double](config.numRuns, edges.length)
    val maxErrors = Array.ofDim[Double](config.numRuns, edges.length)

    //  fm.saveEdgeList(sEdges, "g")
    //    for (ec <- 0 until sEdges.size) {
    //      fm.saveEdgeList(sEdges.splitAt(ec + 1)._1, s"g-e${(ec + 1) * 2}")
    //    }
    for (nr <- 0 until config.numRuns) {
      GraphMap.reset
      WalkStorage.reset
      var prevWalks = ParSeq.empty[(Int, Seq[Int])]

      for (ec <- 0 until sEdges.size) {
        val e = sEdges(ec)
        val result = streamingAddAndRun(e, prevWalks)
        prevWalks = result._1
        val ns = result._2
        val nw = result._3
        numSteps(nr)(ec) = ns
        numWalkers(nr)(ec) = nw
        val nEdges = GraphMap.getNumEdges
        val (meanE, maxE): (Double, Double) = GraphUtils.computeErrorsMeanAndMax(result._1, config)
        meanErrors(nr)(ec) = meanE
        maxErrors(nr)(ec) = maxE
        println(s"Number of edges: ${nEdges}")
        println(s"Number of vertices: ${GraphMap.getNumVertices}")
        println(s"Number of walks: ${prevWalks.size}")
        println(s"Mean Error: ${meanE}")
        println(s"Max Error: ${maxE}")
        //        fm.savePaths(prevWalks, s"${config.rrType.toString}-wl${config.walkLength}-nw${
        //          config.numWalks
        //        }-e${nEdges}-s${e._1.toString}-d${e._2._1.toString}-$nr")

      }
      fm.savePaths(prevWalks, s"${config.rrType.toString}-wl${config.walkLength}-nw${
        config.numWalks
      }-$nr")
    }
    fm.saveComputations(numSteps, Property.stepsToCompute.toString)
    fm.saveComputations(numWalkers, Property.walkersToCompute.toString)
    fm.saveErrors(meanErrors, Property.meanErrors.toString)
    fm.saveErrors(maxErrors, Property.maxErrors.toString)
  }


  def streamingAddAndRun(targetEdge: (Int, (Int, Float)), paths: ParSeq[(Int, Seq[Int])]):
  (ParSeq[(Int, Seq[Int])],
    Int, Int) = {
    val src = targetEdge._1
    val dst = targetEdge._2._1
    val w = targetEdge._2._2
    GraphMap.addUndirectedEdge(src, dst, w)

    println(s"Added Edge: $src <-> $dst")
    val afs1 = mutable.Set(src, dst)
    val afSet = new mutable.HashSet[Int]()
    afSet.add(src)
    afSet.add(dst)

    val result = config.rrType match {
      case RrType.m1 => {
        val init = rwalk.createWalkersByVertices(GraphMap.getVertices().par)
        val ns = computeNumSteps(init)
        val nw = computeNumWalkers(init)
        //        numSteps(i) = Array.fill(vertices.length)(ns)
        //        numWalkers(i) = Array.fill(vertices.length)(nw)
        val p = config.wType match {
          case WalkType.firstorder => rwalk.firstOrderWalk(init)
          case WalkType.secondorder => rwalk.secondOrderWalk(init)
        }
        (p, ns, nw)
      }
      case RrType.m2 => {
        var fWalkers: ParSeq[(Int, (Int, Seq[Int]))] = filterUniqueWalkers(paths, afs1)
        if (paths.filter(_._2.head == src).size == 0) {
          fWalkers ++= ParSeq((src, (1, Seq(src))))
        }
        if (paths.filter(_._2.head == dst).size == 0) {
          fWalkers ++= ParSeq((dst, (1, Seq(dst))))
        }
        val walkers: ParSeq[(Int, (Int, Seq[Int]))] = ParSeq.fill(config.numWalks)(fWalkers).flatten
        val ns = computeNumSteps(walkers)
        val nw = computeNumWalkers(walkers)

        val pp = config.wType match {
          case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
          case WalkType.secondorder => rwalk.secondOrderWalk(walkers)
        }
        val aws = fWalkers.map(tuple => tuple._1).seq
        val up = paths.filter { case (_, p) =>
          !aws.contains(p.head)
        }

        val np = up.union(pp)
        (np, ns, nw)
      }
      case RrType.m3 => {
        val walkers = WalkStorage.filterAffectedPathsForM3(afSet, config)
        //        if (paths.forall(_.head != src)) {
        //          walkers ++= rwalk.initWalker(src)
        //        }
        //        if (paths.forall(_.head != dst)) {
        //          walkers ++= rwalk.initWalker(dst)
        //        }
        val ns = computeNumStepsWithIds(walkers)
        val nw = computeNumWalkersWithIds(walkers)
        //        numSteps(i)(j) = ns
        //        numWalkers(i)(j) = nw
        val partialPaths = config.wType match {
          //          case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
          case WalkType.secondorder => rwalk.secondOrderWalkWitIds(walkers)
        }
        //        val unaffectedPaths = filterUnaffectedPaths(paths, afs1)
        //        val newPaths = unaffectedPaths.union(partialPaths)
        WalkStorage.updatePaths(partialPaths)
        val newPaths = WalkStorage.getPaths()
        (newPaths, ns, nw)
      }
      case RrType.m4 => {
        var walkers = filterWalkers(paths, afs1)
        if (paths.forall(_._2.head != src)) {
          walkers ++= rwalk.initWalker(src)
        }
        if (paths.forall(_._2.head != dst)) {
          walkers ++= rwalk.initWalker(dst)
        }

        val ns = computeNumSteps(walkers)
        val nw = computeNumWalkers(walkers)
        //        numSteps(i)(j) = ns
        //        numWalkers(i)(j) = nw
        val partialPaths = config.wType match {
          case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
          case WalkType.secondorder => rwalk.secondOrderWalk(walkers)
        }
        val unaffectedPaths = filterUnaffectedPaths(paths, afs1)
        val newPaths = unaffectedPaths.union(partialPaths)
        (newPaths, ns, nw)
      }
    }

    result
  }

  //  def streamingCoAuthors(): Unit = {
  //    val edges = fm.readEdgeListByYear()
  //    val numSteps = Array.ofDim[Int](config.numRuns, edges.length)
  //    val numWalkers = Array.ofDim[Int](config.numRuns, edges.length)
  //    val meanErrors = Array.ofDim[Double](config.numRuns, edges.length)
  //    val maxErrors = Array.ofDim[Double](config.numRuns, edges.length)
  //
  //
  //    for (nr <- 0 until config.numRuns) {
  //      GraphMap.reset
  //      var prevWalks = ParSeq.empty[Seq[Int]]
  //
  //      for (ec <- 0 until edges.length) {
  //        val (year, updates) = edges(ec)
  //        val result = streamingAddAndRun(updates, prevWalks)
  //        prevWalks = result._1
  //        val ns = result._2
  //        val nw = result._3
  //
  //        val (meanE, maxE): (Double, Double) = GraphUtils.computeErrorsMeanAndMax(result._1,
  // config)
  //        numSteps(nr)(ec) = ns
  //        numWalkers(nr)(ec) = nw
  //        meanErrors(nr)(ec) = meanE
  //        maxErrors(nr)(ec) = maxE
  //        val nEdges = GraphMap.getNumEdges
  //        println(s"Year: ${year}")
  //        println(s"Number of edges: ${nEdges}")
  //        println(s"Number of vertices: ${GraphMap.getNumVertices}")
  //        println(s"Number of walks: ${prevWalks.size}")
  //        println(s"Mean Error: ${meanE}")
  //        println(s"Max Error: ${maxE}")
  //        println(s"Number of actual steps: $ns")
  //        println(s"Number of actual walks: $nw")
  //        //        fm.savePaths(prevWalks, s"${config.rrType.toString}-wl${config
  // .walkLength}-nw${
  //        //          config.numWalks
  //        //        }-$year")
  //      }
  //      fm.savePaths(prevWalks, s"${config.rrType.toString}-wl${config.walkLength}-nw${
  //        config.numWalks
  //      }-final")
  //    }
  //    fm.saveComputations(numSteps, Property.stepsToCompute.toString)
  //    fm.saveComputations(numWalkers, Property.walkersToCompute.toString)
  //    fm.saveErrors(meanErrors, Property.meanErrors.toString)
  //    fm.saveErrors(maxErrors, Property.maxErrors.toString)
  //  }
  def streamingCoAuthors(): Unit = {
    val edges = fm.readEdgeListByYear()
    val numSteps = Array.ofDim[Int](config.numRuns, edges.length)
    val numWalkers = Array.ofDim[Int](config.numRuns, edges.length)
    val meanErrors = Array.ofDim[Double](config.numRuns, edges.length)
    val maxErrors = Array.ofDim[Double](config.numRuns, edges.length)


    for (nr <- 0 until config.numRuns) {
      GraphMap.reset
      WalkStorage.reset
      var prevWalks = ParSeq.empty[(Int, Seq[Int])]

      for (ec <- 0 until edges.length) {
        val (year, updates) = edges(ec)
        val result = streamingAddAndRunWithId(updates, prevWalks)
        prevWalks = result._1
        val ns = result._2
        val nw = result._3

        val (meanE, maxE): (Double, Double) = GraphUtils.computeErrorsMeanAndMax(result._1, config)
        numSteps(nr)(ec) = ns
        numWalkers(nr)(ec) = nw
        meanErrors(nr)(ec) = meanE
        maxErrors(nr)(ec) = maxE
        val nEdges = GraphMap.getNumEdges
        println(s"Year: ${year}")
        println(s"Number of edges: ${nEdges}")
        println(s"Number of vertices: ${GraphMap.getNumVertices}")
        println(s"Number of walks: ${prevWalks.size}")
        println(s"Mean Error: ${meanE}")
        println(s"Max Error: ${maxE}")
        println(s"Number of actual steps: $ns")
        println(s"Number of actual walks: $nw")
        //        fm.savePaths(prevWalks, s"${config.rrType.toString}-wl${config.walkLength}-nw${
        //          config.numWalks
        //        }-$year")
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

  def streamEdges(): Unit = {
    val edges = fm.readPartitionedEdgeList()
    val numSteps = Array.ofDim[Int](config.numRuns, edges.length)
    val numWalkers = Array.ofDim[Int](config.numRuns, edges.length)
    val meanErrors = Array.ofDim[Double](config.numRuns, edges.length)
    val maxErrors = Array.ofDim[Double](config.numRuns, edges.length)


    for (nr <- 0 until config.numRuns) {
      GraphMap.reset
      WalkStorage.reset
      var prevWalks = ParSeq.empty[(Int, Seq[Int])]

      for (ec <- 0 until edges.length) {
        val (year, updates) = edges(ec)
        val result = streamingAddAndRunWithId(updates, prevWalks)
        prevWalks = result._1
        val ns = result._2
        val nw = result._3

        val (meanE, maxE): (Double, Double) = GraphUtils.computeErrorsMeanAndMax(result._1, config)
        numSteps(nr)(ec) = ns
        numWalkers(nr)(ec) = nw
        meanErrors(nr)(ec) = meanE
        maxErrors(nr)(ec) = maxE
        val nEdges = GraphMap.getNumEdges
        println(s"Year: ${year}")
        println(s"Number of edges: ${nEdges}")
        println(s"Number of vertices: ${GraphMap.getNumVertices}")
        println(s"Number of walks: ${prevWalks.size}")
        println(s"Mean Error: ${meanE}")
        println(s"Max Error: ${maxE}")
        println(s"Number of actual steps: $ns")
        println(s"Number of actual walks: $nw")
        //        fm.savePaths(prevWalks, s"${config.rrType.toString}-wl${config.walkLength}-nw${
        //          config.numWalks
        //        }-$year")
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
        val ns = computeNumSteps(init)
        val nw = computeNumWalkers(init)
        val p = config.wType match {
          //          case WalkType.firstorder => rwalk.firstOrderWalk(init)
          case WalkType.secondorder => rwalk.secondOrderWalk(init)
        }
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
        val nw = computeNumWalkers(walkers)

        val pp = config.wType match {
          //          case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
          case WalkType.secondorder => rwalk.secondOrderWalk(walkers)
        }
        val aws = fWalkers.map(tuple => tuple._1).seq
        val up = paths.filter { case (wVersion, p) =>
          !aws.contains(p.head)
        }

        val np = up.union(pp)
        (np, ns, nw)
      }
      case RrType.m3 => {
        val walkers = WalkStorage.filterAffectedPathsForM3(afs, config)

        val ns = computeNumStepsWithIds(walkers)
        val nw = computeNumWalkersWithIds(walkers)
        val partialPaths = config.wType match {
          //          case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
          case WalkType.secondorder => rwalk.secondOrderWalkWitIds(walkers)
        }
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
        val nw = computeNumWalkers(walkers)
        val partialPaths = config.wType match {
          //          case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
          case WalkType.secondorder => rwalk.secondOrderWalk(walkers)
        }
        val unaffectedPaths = filterUnaffectedPaths(paths, afs)
        val newPaths = unaffectedPaths.union(partialPaths)
        (newPaths, ns, nw)
      }
    }

    result
  }

  def streamingAddAndRun(updates: ParSeq[(Int, Int, Int)], paths: ParSeq[(Int, Seq[Int])]):
  (ParSeq[(Int, Seq[Int])],
    Int, Int) = {
    val afs = new mutable.HashSet[Int]()
    for (u <- updates.seq) {
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
        val ns = computeNumSteps(init)
        val nw = computeNumWalkers(init)
        val p = config.wType match {
          case WalkType.firstorder => rwalk.firstOrderWalk(init)
          case WalkType.secondorder => rwalk.secondOrderWalk(init)
        }
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
        val nw = computeNumWalkers(walkers)

        val pp = config.wType match {
          case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
          case WalkType.secondorder => rwalk.secondOrderWalk(walkers)
        }
        val aws = fWalkers.map(tuple => tuple._1).seq
        val up = paths.filter { case (_, p) =>
          !aws.contains(p.head)
        }

        val np = up.union(pp)
        (np, ns, nw)
      }
      case RrType.m3 => {
        var walkers = filterSplitPaths(paths, afs)
        for (a <- afs) {
          if (walkers.count(_._1 == a) == 0) {
            walkers ++= rwalk.initWalker(a)
          }
        }
        val ns = computeNumSteps(walkers)
        val nw = computeNumWalkers(walkers)
        val partialPaths = config.wType match {
          case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
          case WalkType.secondorder => rwalk.secondOrderWalk(walkers)
        }
        val unaffectedPaths = filterUnaffectedPaths(paths, afs)
        val newPaths = unaffectedPaths.union(partialPaths)
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
        val nw = computeNumWalkers(walkers)
        val partialPaths = config.wType match {
          case WalkType.firstorder => rwalk.firstOrderWalk(walkers)
          case WalkType.secondorder => rwalk.secondOrderWalk(walkers)
        }
        val unaffectedPaths = filterUnaffectedPaths(paths, afs)
        val newPaths = unaffectedPaths.union(partialPaths)
        (newPaths, ns, nw)
      }
    }

    result
  }


  def removeAndRun(): Unit = {
    val g1 = fm.readFromFile(config.directed)
    val vertices: Seq[Int] = g1.map(_._1).seq
    for (target <- vertices) {
      println(s"Removed vertex $target")
      val g2 = removeVertex(g1, target)
      val init = rwalk.initRandomWalk(g2)
      for (i <- 0 until config.numRuns) {
        val seed = System.currentTimeMillis()
        val r = new Random(seed)
        val paths = rwalk.firstOrderWalk(init, nextFloat = r.nextFloat)
        fm.savePaths(paths, s"v${
          target.toString
        }-$i-s$seed")
      }
    }
  }

  def removeVertex(g: ParSeq[(Int, mutable.Set[(Int, Float)])], target: Int): ParSeq[(Int,
    mutable.Set[(Int,
    Float)])] = {
    g.filter(_._1 != target).map {
      case (vId, neighbors) =>
        val filtered = neighbors.filter(_._1 != target)
        (vId, filtered)
    }
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

  def computeNumWalkers(walkers: ParSeq[(Int, (Int, Seq[Int]))]) = {
    walkers.length
  }

  def computeNumWalkersWithIds(walkers: ParSeq[(Int, (Int, Seq[Int]))]) = {
    walkers.length
  }

  def streamingAffecteds() = {
    for (nr <- 0 until config.numRuns) {
      GraphMap.reset
      val walks = rwalk.secondOrderWalk(rwalk.loadGraph())
      val numOccurs = walks.flatMap { case (wVersion, w) =>
        w.distinct.map { case v => (v, 1) }
      }.groupBy(_._1).map { case (v, tuple) =>
        val occurs: Int = tuple.foldLeft(0)(_ + _._2)
        (v, occurs)
      }.toSeq.seq.sortWith(_._2 < _._2)
      fm.saveCounts(numOccurs, s"${Property.countsSuffix}-$nr")
    }
  }
}
