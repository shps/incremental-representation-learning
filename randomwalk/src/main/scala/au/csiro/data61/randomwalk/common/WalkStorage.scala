package au.csiro.data61.randomwalk.common

import java.util
import java.util.Map
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

import au.csiro.data61.randomwalk.common.CommandParser.WalkType

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.parallel.ParSeq

/**
  * Created by Hooman on 2018-03-15.
  */
object WalkStorage {

  val idCounter = new AtomicInteger(0)
  val walkMap = new ConcurrentHashMap[Int, (Int, Int, Seq[Int])]()
  val vertexWalkMap = new ConcurrentHashMap[Int, ConcurrentHashMap[Int, Int]]()


  def updatePaths(partialPaths: ParSeq[(Int, (Int, Int, Seq[Int]))]) = {
    println("****** Updating WalkStorage ******")
    partialPaths.foreach { case walk =>
      val walkId = walk._1
      val wVersion = walk._2._1 + 1
      val firstIndex = walk._2._2
      walkMap.put(walkId, (wVersion, firstIndex, walk._2._3))
      walk._2._3.foreach { case v =>
        vertexWalkMap.computeIfAbsent(v, _ => new ConcurrentHashMap[Int, Int]()).put(walkId,
          wVersion)
      }
    }
  }


  def getPaths(): ParSeq[(Int, Int, Seq[Int])] = {
    //    walkMap.toMap.par.values.toSeq
    walkMap.values.toSeq.par
  }

  def getPathsWithIds(): ConcurrentHashMap[Int, (Int, Int, Seq[Int])] = {
    return walkMap
  }

  def walkSize(): Int = {
    walkMap.size()
  }

  def filterAffectedPathsForM3(afs: mutable.HashSet[Int], config: Params) = {
    println("****** WalkStorage: Filter Affected Paths ******")
    val maxLength = config.wType match {
      case WalkType.firstorder => config.walkLength + 1
      case WalkType.secondorder => config.walkLength + 2
    }

    filterAffectedPaths(afs, config).map { case (wId, duplicates) =>
      val wVersion = duplicates.head._2._1
      val walk = duplicates.head._2._3
      val first = walk.indexWhere(e => afs.contains(e)) + 1
      if (first == 0) {
        print("Something is wrong. It can find the affected index in the path that already has " +
          "the affected vertex!!!")

      }
      (wId, (wVersion, first, walk.splitAt(first)._1))
      // do not consider walks that only their last element is affected
    }.filter(_._2._2 < maxLength).toSeq
  }

  def filterAffectedPathsForM4(afs: mutable.HashSet[Int], config: Params) = {
    println("****** WalkStorage: Filter Affected Paths ******")

    filterAffectedPaths(afs, config).map { case (wId, duplicates) =>
      val wVersion = duplicates.head._2._1
      val walk = duplicates.head._2._3
      (wId, (wVersion, 0, Seq(walk.head)))
    }.toSeq
  }

  def filterAffectedPaths(afs: mutable.HashSet[Int], config: Params) = {
    afs.par.flatMap { case v =>
      val ws = vertexWalkMap.get(v)

      if (ws == null) {
        Seq.tabulate(config.numWalks)(_ => {
          Seq((idCounter.incrementAndGet(), (0, 0, Seq(v))))
        }).flatten
      } else {
        var allWalks = Seq.empty[(Int, (Int, Int, Seq[Int]))]
        val it = ws.entrySet.iterator
        while (it.hasNext) {
          val next = it.next()
          val (wId, oldVersion) = (next.getKey, next.getValue)
          val (latestVersion, firstIndex, w) = walkMap.get(wId)
          if (w == null || w.isEmpty) {
            println("Something is wrong!!! w is null!!")
          } else if (latestVersion == oldVersion) { // vertex-index exists in the path
            allWalks ++= Seq((wId, (latestVersion, firstIndex, w)))
          } else {
            it.remove() // vertex-index is old and needs to be removed.
          }
        }

        allWalks
      }
    }.groupBy(_._1)
  }

  def reset: Unit = {
    idCounter.set(0)
    walkMap.clear()
    vertexWalkMap.clear()
  }
}
