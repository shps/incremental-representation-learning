package au.csiro.data61.randomwalk.common

import scala.collection.parallel.ParSeq
import scala.util.Random

/**
  * Created by Hooman on 2018-04-09.
  */
object Word2VecUtils {

  /**
    * Generates a sequence of target-context pairs.
    * @param walks
    * @param numSkips
    * @param window
    * @return
    */
  def createPairs(walks: ParSeq[Seq[Int]], numSkips: Int, window: Int): ParSeq[(Int, Int)] = {
    walks.flatMap { case walk =>
      var pairs = Seq.empty[(Int, Int)]
      for (index <- 0 until walk.length) {
        var currentPairs = Seq.empty[(Int, Int)]
        val left = math.max(index - window, 0)
        val right = math.min(index + window, walk.length - 1)
        for (i <- left until index) {
          currentPairs ++= Seq((walk(index), walk(i)))
        }
        for (i <- right until index by -1) {
          currentPairs ++= Seq((walk(index), walk(i)))
        }
        pairs ++= Random.shuffle(currentPairs).take(numSkips)
      }
      pairs
    }
  }

  def createVocabulary(walks: ParSeq[Seq[Int]]): ParSeq[Int] = {
    walks.flatten.distinct.seq.sortWith(_ < _).par
  }
}
