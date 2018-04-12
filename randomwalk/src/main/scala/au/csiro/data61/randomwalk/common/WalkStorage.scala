package au.csiro.data61.randomwalk.common

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.parallel.ParSeq

/**
  * Created by Hooman on 2018-03-15.
  */
object WalkStorage {

  val idCounter = new AtomicInteger(0)
  val walkMap = new ConcurrentHashMap[Int, (Int, Seq[Int])]()
  val vertexWalkMap = new ConcurrentHashMap[Int, ConcurrentHashMap[Int, Int]]()

  //  def getNewAndRemovedVertices(prevWalk: Seq[Int], walk: Seq[Int]):
  //  (mutable.Set[Int], mutable.Set[Int]) = {
  //    val prevSet = prevWalk.toSet
  //    val added = mutable.Set(walk: _*)
  //    val removed = mutable.Set[Int]()
  //    for (pw <- prevSet) {
  //      if (!added.contains(pw)) {
  //        removed.add(pw)
  //      }
  //    }
  //
  //    (added, removed)
  //  }

  def updatePaths(partialPaths: ParSeq[(Int, (Int, Seq[Int]))]) = {
    println("****** Updating WalkStorage ******")
    partialPaths.foreach { case walk =>
      val walkId = walk._1
      //      var prevWalk = walkMap.getOrDefault(walkId, (0, Seq[Int]()))
      //      val (added, removed) = getNewAndRemovedVertices(prevWalk, walk._2)
      val wVersion = walk._2._1 + 1
      walkMap.put(walkId, (wVersion, walk._2._2))
      // Needs improvement
      //      removed.foreach { case v =>
      //        vertexWalkMap.computeIfAbsent(v, _ => new ConcurrentHashMap[Int, Unit]()).remove
      // (walkId)
      //      }
      //      added.foreach { case v =>
      //        vertexWalkMap.computeIfAbsent(v, _ => new ConcurrentHashMap[Int, Unit]()).put
      // (walkId,
      //          null)
      walk._2._2.foreach { case v =>
        vertexWalkMap.computeIfAbsent(v, _ => new ConcurrentHashMap[Int, Int]()).put(walkId,
          wVersion)
      }
    }
  }


  def getPaths(): ParSeq[(Int, Seq[Int])] = {
    walkMap.toMap.par.values.toSeq
  }

  def walkSize(): Int = {
    walkMap.size()
  }

  def filterAffectedPaths(afs: mutable.HashSet[Int], config: Params) = {
    println("****** WalkStorage: Filter Affected Paths ******")
    //    afs.par.flatMap { case v =>
    //      val ws = vertexWalkMap.get(v)
    //
    //      if (ws == null) {
    //        Seq.tabulate(config.numWalks)(_ => {
    //          Seq((v, (idCounter.incrementAndGet(), Seq(v))))
    //        }).flatten
    //      } else {
    //        var allWalks = Seq.empty[(Int, (Int, Seq[Int]))]
    //        for (id <- ws.keys()) {
    //          val w = walkMap.get(id)
    //          if (w == null || w.isEmpty) {
    //            println("Something is wrong!!! w is null!!")
    //          }
    //          val first = w.indexWhere(e => afs.contains(e))
    //          allWalks ++= Seq((w.head, (id, w.splitAt(first + 1)._1)))
    //        }
    //        allWalks
    //      }
    //    }.groupBy(_._2._1).map { case (v, duplicates) => (v, duplicates.head._2) }.toSeq
    afs.par.flatMap { case v =>
      val ws = vertexWalkMap.get(v)

      if (ws == null) {
        Seq.tabulate(config.numWalks)(_ => {
          Seq((idCounter.incrementAndGet(), (0, Seq(v))))
        }).flatten
      } else {
        var allWalks = Seq.empty[(Int, (Int, Seq[Int]))]
        val it = ws.entrySet.iterator
        while (it.hasNext) {
          val next = it.next()
          val (wId, oldVersion) = (next.getKey, next.getValue)
          val (latestVersion, w) = walkMap.get(wId)
          if (w == null || w.isEmpty) {
            println("Something is wrong!!! w is null!!")
          } else if (latestVersion == oldVersion) { // vertex-index exists in the path
            allWalks ++= Seq((wId, (latestVersion, w)))
          } else {
            it.remove() // vertex-index is old and needs to be removed.
          }
        }
        //        for (id <- ws.keys()) {
        //          val (wVersion, w) = walkMap.get(id)
        //          if (w == null || w.isEmpty) {
        //            println("Something is wrong!!! w is null!!")
        //          }
        //          allWalks ++= Seq((id, w))
        //        }
        allWalks
      }
    }.groupBy(_._1).map { case (wId, duplicates) =>
      val wVersion = duplicates.head._2._1
      val walk = duplicates.head._2._2
      val first = walk.indexWhere(e => afs.contains(e))
      (wId, (wVersion, walk.splitAt(first + 1)._1))
    }.toSeq
  }

  def reset: Unit = {
    idCounter.set(0)
    walkMap.clear()
    vertexWalkMap.clear()
  }
}
