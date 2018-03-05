package au.csiro.data61.randomwalk.algorithm

import au.csiro.data61.randomwalk.common.CommandParser.{RrType, WalkType}
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
    val g1 = fm.readFromFile(config.directed)
    val vertices: Seq[Int] = g1.map(_._1).seq.sortWith((a, b) => a < b)
    val numSteps = Array.ofDim[Int](config.numRuns, vertices.length)
    val numWalkers = Array.ofDim[Int](config.numRuns, vertices.length)

    for (i <- 0 until config.numRuns) {
      for (j <- 0 until math.min(config.numVertices, vertices.length)) {
        val target = vertices(j)
        val g2 = removeVertex(g1, target)
        val init = rwalk.initRandomWalk(g2)
        val paths = config.wType match {
          case WalkType.firstorder => rwalk.firstOrderWalk(init)
          case WalkType.secondorder => rwalk.secondOrderWalk(init)
        }
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
            val p = config.wType match {
              case WalkType.firstorder => rwalk.firstOrderWalk(init)
              case WalkType.secondorder => rwalk.secondOrderWalk(init)
            }
            p
          }
          case RrType.m2 => {
            val fWalkers: Seq[(Int, Seq[Int])] = filterUniqueWalkers(
              paths, afs1) ++ Seq((target, Seq(target)))
            val walkers: Seq[(Int, Seq[Int])] = Seq.fill(config
              .numWalks)(fWalkers).flatMap(a => a)
            val ns = computeNumSteps(walkers)
            val nw = computeNumWalkers(walkers)
            numSteps(i)(j) = ns
            numWalkers(i)(j) = nw
            val pp = config.wType match {
              case WalkType.firstorder => rwalk.firstOrderWalk(init)
              case WalkType.secondorder => rwalk.secondOrderWalk(init)
            }
            val aws = fWalkers.map(tuple => tuple._1).seq
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
            val partialPaths = config.wType match {
              case WalkType.firstorder => rwalk.firstOrderWalk(init)
              case WalkType.secondorder => rwalk.secondOrderWalk(init)
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
              case WalkType.firstorder => rwalk.firstOrderWalk(init)
              case WalkType.secondorder => rwalk.secondOrderWalk(init)
            }
            val unaffectedPaths = filterUnaffectedPaths(paths, afs1)
            val newPaths = unaffectedPaths.union(partialPaths)
            newPaths
          }
        }

        fm.savePaths(newPaths.seq, s"${config.rrType.toString}-wl${config.walkLength}-nw${
          config
            .numWalks
        }-v${target.toString}-$i")
      }

    }


    fm.saveNumSteps(vertices, numSteps, Property.stepsToCompute.toString)
    fm.saveNumSteps(vertices, numWalkers, Property.walkersToCompute.toString)
  }

  def extractEdges(g1: Seq[(Int, Seq[(Int, Float)])]): Seq[(Int, (Int, Float))] = {
    g1.flatMap { case (src, neighbors) =>
      neighbors.map { case (dst, w) => (src, (dst, w)) }
    }
  }

  def streamingUpdates(): Unit = {
    val g1 = fm.readFromFile(directed = true) // read it as directed
    val edges: Seq[(Int, (Int, Float))] = extractEdges(g1)
    print(s"Number of edges: ${edges.length}")
    val rand = new Random(config.seed)
    val sEdges = rand.shuffle(edges.seq)
    val numSteps = Array.ofDim[Int](config.numRuns, sEdges.length)
    val numWalkers = Array.ofDim[Int](config.numRuns, sEdges.length)

    fm.saveEdgeList(sEdges.seq, "g")
    //    for (ec <- 0 until sEdges.size) {
    //      fm.saveEdgeList(sEdges.splitAt(ec + 1)._1, s"g-e${(ec + 1) * 2}")
    //    }
    for (nr <- 0 until config.numRuns) {
      GraphMap.reset
      var prevWalks = Seq.empty[Seq[Int]]

      for (ec <- 0 until sEdges.size) {
        val e = sEdges(ec)
        val result = streamingAddAndRun(e, prevWalks)
        prevWalks = result._1
        val ns = result._2
        val nw = result._3
        numSteps(nr)(ec) = ns
        numWalkers(nr)(ec) = nw
        val nEdges = GraphMap.getNumEdges
        println(s"Number of edges: ${nEdges}")
        println(s"Number of vertices: ${GraphMap.getNumVertices}")
        println(s"Number of walks: ${prevWalks.size}")
        //        fm.savePaths(prevWalks, s"${config.rrType.toString}-wl${config.walkLength}-nw${
        //          config.numWalks
        //        }-e${nEdges}-s${e._1.toString}-d${e._2._1.toString}-$nr")

      }
      fm.savePaths(prevWalks.seq, s"${config.rrType.toString}-wl${config.walkLength}-nw${
        config.numWalks
      }-$nr")
    }
    fm.saveComputations(numSteps, Property.stepsToCompute.toString)
    fm.saveComputations(numWalkers, Property.walkersToCompute.toString)
  }

  def streamingAddAndRun(targetEdge: (Int, (Int, Float)), paths: Seq[Seq[Int]]): (Seq[Seq[Int]],
    Int, Int) = {
    val src = targetEdge._1
    val dst = targetEdge._2._1
    val w = targetEdge._2._2
    var sNeighbors = GraphMap.getNeighbors(src)
    var dNeighbors = GraphMap.getNeighbors(dst)
    sNeighbors ++= Seq((dst, w))
    dNeighbors ++= Seq((src, w))
    GraphMap.putVertex(src, sNeighbors)
    GraphMap.putVertex(dst, dNeighbors)

    println(s"Added Edge: $src <-> $dst")
    val afs1 = Seq(src, dst)

    val result = config.rrType match {
      case RrType.m1 => {
        val init = rwalk.createWalkersByVertices(GraphMap.getVertices())
        val ns = computeNumSteps(init)
        val nw = computeNumWalkers(init)
        //        numSteps(i) = Array.fill(vertices.length)(ns)
        //        numWalkers(i) = Array.fill(vertices.length)(nw)
        val p = rwalk.firstOrderWalk(init)
        (p, ns, nw)
      }
      case RrType.m2 => {
        var fWalkers: Seq[(Int, Seq[Int])] = filterUniqueWalkers(paths, afs1)
        if (paths.forall(_.head != src)) {
          fWalkers ++= Seq((src, Seq(src)))
        }
        if (paths.forall(_.head != dst)) {
          fWalkers ++= Seq((dst, Seq(dst)))
        }
        val walkers: Seq[(Int, Seq[Int])] = Seq.fill(config.numWalks)(fWalkers).flatMap(a => a)
        val ns = computeNumSteps(walkers)
        val nw = computeNumWalkers(walkers)

        val pp = rwalk.firstOrderWalk(walkers)
        val aws = fWalkers.map(tuple => tuple._1).seq
        val up = paths.filter { case p =>
          !aws.contains(p.head)
        }

        val np = up.union(pp)
        (np, ns, nw)
      }
      case RrType.m3 => {
        var walkers = filterSplitPaths(paths, afs1)
        if (paths.forall(_.head != src)) {
          walkers ++= rwalk.initWalker(src)
        }
        if (paths.forall(_.head != dst)) {
          walkers ++= rwalk.initWalker(dst)
        }
        val ns = computeNumSteps(walkers)
        val nw = computeNumWalkers(walkers)
        //        numSteps(i)(j) = ns
        //        numWalkers(i)(j) = nw
        val partialPaths = rwalk.firstOrderWalk(walkers)
        val unaffectedPaths = filterUnaffectedPaths(paths, afs1)
        val newPaths = unaffectedPaths.union(partialPaths)
        (newPaths, ns, nw)
      }
      case RrType.m4 => {
        var walkers = filterWalkers(paths, afs1)
        if (paths.forall(_.head != src)) {
          walkers ++= rwalk.initWalker(src)
        }
        if (paths.forall(_.head != dst)) {
          walkers ++= rwalk.initWalker(dst)
        }

        val ns = computeNumSteps(walkers)
        val nw = computeNumWalkers(walkers)
        //        numSteps(i)(j) = ns
        //        numWalkers(i)(j) = nw
        val partialPaths = rwalk.firstOrderWalk(walkers)
        val unaffectedPaths = filterUnaffectedPaths(paths, afs1)
        val newPaths = unaffectedPaths.union(partialPaths)
        (newPaths, ns, nw)
      }
    }

    result
  }


  def removeAndRun(): Unit = {
    val g1 = fm.readFromFile(config.directed)
    val vertices: Seq[Int] = g1.map(_._1)
    for (target <- vertices) {
      println(s"Removed vertex $target")
      val g2 = removeVertex(g1, target)
      val init = rwalk.initRandomWalk(g2)
      for (i <- 0 until config.numRuns) {
        val seed = System.currentTimeMillis()
        val r = new Random(seed)
        val paths = rwalk.firstOrderWalk(init, nextFloat = r.nextFloat)
        fm.savePaths(paths.seq, s"v${
          target.toString
        }-$i-s$seed")
      }
    }
  }

  def removeVertex(g: Seq[(Int, Seq[(Int, Float)])], target: Int): Seq[(Int, Seq[(Int,
    Float)])] = {
    g.filter(_._1 != target).map {
      case (vId, neighbors) =>
        val filtered = neighbors.filter(_._1 != target)
        (vId, filtered)
    }
  }

  def filterUniqueWalkers(paths: Seq[Seq[Int]], afs1: Seq[Int]) = {
    filterWalkers(paths, afs1).groupBy(_._1).map {
      case (_, p) =>
        p.head
    }.toSeq
  }

  def filterWalkers(paths: Seq[Seq[Int]], afs1: Seq[Int]): Seq[(Int, Seq[Int])] = {
    filterAffectedPaths(paths, afs1).map(a => (a.head, Seq(a.head)))
  }

  def filterSplitPaths(paths: Seq[Seq[Int]], afs1: Seq[Int]) = {
    filterAffectedPaths(paths, afs1).map {
      a =>
        val first = a.indexWhere(e => afs1.contains(e))
        (a.head, a.splitAt(first + 1)._1)
    }
  }

  def filterAffectedPaths(paths: Seq[Seq[Int]], afs1: Seq[Int]) = {
    paths.filter {
      case p =>
        !p.forall(s => !afs1.contains(s))
    }
  }

  def filterUnaffectedPaths(paths: Seq[Seq[Int]], afs1: Seq[Int]) = {
    paths.filter {
      case p =>
        p.forall(s => !afs1.contains(s))
    }
  }

  def computeNumSteps(walkers: Seq[(Int, Seq[Int])]) = {
    val bcWalkLength = config.walkLength + 1
    walkers.map {
      case (_, path) => bcWalkLength - path.length
    }.reduce(_ + _)
  }

  def computeNumWalkers(walkers: Seq[(Int, Seq[Int])]) = {
    walkers.length
  }
}
