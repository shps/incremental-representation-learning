package au.csiro.data61.randomwalk

import au.csiro.data61.randomwalk.common.CommandParser.TaskName
import au.csiro.data61.randomwalk.common._
import au.csiro.data61.randomwalk.experiments.StreamingExperiment
import org.apache.log4j.LogManager

import scala.collection.parallel.ParSeq
import scala.util.Random

object Main {
  lazy val logger = LogManager.getLogger("myLogger")

  def main(args: Array[String]) {
    CommandParser.parse(args) match {
      case Some(params) =>
        execute(params)
      case None => sys.exit(1)
    }
  }

  def execute(params: Params): Unit = {
    params.cmd match {
      case TaskName.sca =>
        StreamingExperiment(params).streamEdges()
      case TaskName.gPairs =>
        println("Reading the walks...")
        val fm = FileManager(params)
        Random.setSeed(params.seed)
        val WALK_INDEX = 3
        val walks = params.allWalks match {
          case true => fm.readWalks().map { w => w.splitAt(WALK_INDEX)._2 }
          case false =>
            val newAndOldWalks = fm.readWalks().map { case w =>
              (w(WALK_INDEX - 1), w.splitAt(WALK_INDEX)._2)
            }
              .groupBy(_._1)
            var walks = newAndOldWalks.getOrElse(1, ParSeq.empty[(Int, Seq[Int])])
            val numNewWalks = walks.size
            val takeFromOldWalks = (numNewWalks * params.O).toInt
            println(s"Number of new walks: $numNewWalks")
            println(s"Trying to take $takeFromOldWalks number of old walks...")
            walks = walks.union(Random.shuffle(newAndOldWalks.
              getOrElse(0, ParSeq.empty[(Int, Seq[Int])]).seq).take(takeFromOldWalks)).par
            println(s"Number of old walks taken: ${walks.size - numNewWalks}")
            walks.map(_._2)
        }
        println(s"Generating pairs from ${walks.size} number of walks...")
        val pairs = Word2VecUtils.createPairs(walks, numSkips = params.w2vSkipSize, window =
          params.w2vWindow, params.selfContext, params.forceSkipSize)
        println(s"Generated ${pairs.size} pairs. ForceSkipSize: ${params.forceSkipSize.toString}")
        println("Extracting the vocabulary...")
        val vocab = Word2VecUtils.createVocabulary(walks)
        println("Writing to the file...")
        fm.saveTargetContextPairs(pairs, vocab, s"w${params.w2vWindow}-s${params.w2vSkipSize}")
        println("Completed!")
    }
  }
}
