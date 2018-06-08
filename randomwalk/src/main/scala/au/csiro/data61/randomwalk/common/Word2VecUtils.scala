package au.csiro.data61.randomwalk.common

import scala.collection.parallel.ParSeq
import scala.util.Random

/**
  * Created by Hooman on 2018-04-09.
  */
object Word2VecUtils {

  /**
    * Generates a sequence of target-context pairs.
    *
    * @param walks
    * @param numSkips
    * @param window
    * @return
    */
  def createPairs(walks: ParSeq[Seq[Int]], numSkips: Int, window: Int, selfContext: Boolean,
                  forceSkipSize: Boolean):
  ParSeq[(Int, Int)] = {
    walks.flatMap { case walk =>
      var allPairs = Seq.empty[(Int, Int)]
      for (index <- 0 until walk.length) {
        val target = walk(index)
        var currentPairs = Seq.empty[(Int, Int)]
        val left = math.max(index - window, 0)
        val right = math.min(index + window, walk.length - 1)
        for (i <- left until index) {
          if (target == walk(i) && !selfContext) {
            // Ignore target == context
          }
          else {
            currentPairs ++= Seq((target, walk(i)))
          }
        }
        for (i <- right until index by -1) {
          if (target == walk(i) && !selfContext) {
            // Ignore target == context
          }
          else {
            currentPairs ++= Seq((target, walk(i)))
          }
        }
        if (currentPairs.isEmpty){
          println(s"CurrentPairs is empty for target $target. Make it self-context.")
        }
        var pairs = Seq.empty[(Int, Int)]
        pairs ++= Random.shuffle(currentPairs).take(numSkips)

        if (forceSkipSize) {
          var size = pairs.size
          while (size < numSkips) {
            val numSamples = numSkips - size
            pairs ++= Random.shuffle(currentPairs).take(numSamples)
            size = pairs.size
          }
        }

        allPairs ++= pairs

      }
      allPairs
    }
  }

  def createVocabulary(walks: ParSeq[Seq[Int]]): ParSeq[Int] = {
    walks.flatten.distinct.seq.sortWith(_ < _).par
  }
}
