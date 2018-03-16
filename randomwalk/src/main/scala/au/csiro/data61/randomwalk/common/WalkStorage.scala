package au.csiro.data61.randomwalk.common

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.{AtomicInteger, LongAdder}

import scala.collection.mutable
import scala.collection.parallel.ParSeq
import scala.collection.JavaConversions._

/**
  * Created by Hooman on 2018-03-15.
  */
object WalkStorage {

  def getNewAndRemovedVertices(prevWalk: Seq[Int], walk: Seq[Int]):
  (mutable.Set[Int], mutable.Set[Int]) = {
    val prevSet = prevWalk.toSet
    val added = mutable.Set(walk: _*)
    val removed = mutable.Set[Int]()
    for (pw <- prevSet) {
      if (!added.contains(pw)) {
        removed.add(pw)
      }
    }

    (added, removed)
  }

  def updatePaths(partialPaths: ParSeq[(Int, Seq[Int])]) = {
    println("****** Updating WalkStorage ******")
    partialPaths.foreach { case walk =>
      val walkId = walk._1
      val prevWalk = walkMap.getOrDefault(walkId, Seq[Int]())
      val (added, removed) = getNewAndRemovedVertices(prevWalk, walk._2)
      walkMap.put(walkId, walk._2)
      // Needs improvement
      removed.foreach { case v =>
        vertexWalkMap.computeIfAbsent(v, _ => new ConcurrentHashMap[Int, Unit]()).remove(walkId)
      }
      added.foreach { case v =>
        vertexWalkMap.computeIfAbsent(v, _ => new ConcurrentHashMap[Int, Unit]()).put(walkId,
          null)
      }
    }
  }


  def getPaths(): ParSeq[Seq[Int]] = {
    walkMap.toMap.par.values.toSeq
  }


  val idCounter = new AtomicInteger(0)
  val walkMap = new ConcurrentHashMap[Int, Seq[Int]]()
  val vertexWalkMap = new ConcurrentHashMap[Int, ConcurrentHashMap[Int, Unit]]()

  def walkSize(): Int = {
    walkMap.size()
  }

  def filterAffectedPaths(afs: mutable.HashSet[Int], config: Params) = {
    println("****** WalkStorage: Filter Affected Paths ******")
    afs.par.flatMap { case v =>
      val ws = vertexWalkMap.get(v)

      if (ws == null) {
        Seq.tabulate(config.numWalks)(_ => {
          Seq((v, (idCounter.incrementAndGet(), Seq(v))))
        }).flatten
      } else {
        var allWalks = Seq.empty[(Int, (Int, Seq[Int]))]
        for (id <- ws.keys()) {
          val w = walkMap.get(id)
          if (w == null || w.isEmpty) {
            println("Whast the fik±")
          }
          val first = w.indexWhere(e => afs.contains(e))
          allWalks ++= Seq((w.head, (id, w.splitAt(first + 1)._1)))
        }
        allWalks
      }
    }.groupBy(_._2._1).map { case (v, duplicates) => (v, duplicates.head._2) }.toSeq
  }

  def reset: Unit = {
    idCounter.set(0)
    walkMap.clear()
    vertexWalkMap.clear()
  }
}
