package au.csiro.data61.randomwalk

import au.csiro.data61.randomwalk.algorithm.UniformRandomWalk
import au.csiro.data61.randomwalk.common.CommandParser.TaskName
import au.csiro.data61.randomwalk.common._
import au.csiro.data61.randomwalk.experiments.StreamingExperiment
import org.apache.log4j.LogManager

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
    val rw = UniformRandomWalk(params)
    val fm = FileManager(params)
    //    val exp = Experiments(params)
    val paths = params.cmd match {
      case TaskName.firstorder =>
        val g = rw.loadGraph()
        fm.savePaths(rw.firstOrderWalk(g))
      case TaskName.queryPaths =>
      //        context.textFile(params.input).repartition(params.rddPartitions).
      //          map(_.split("\\s+").map(s => s.toInt))
      case TaskName.probs =>
        val g = rw.loadGraph()
        fm.savePaths(rw.firstOrderWalk(g))
      case TaskName.degrees =>
        rw.loadGraph()
        fm.saveDegrees(GraphUtils.degrees(), s"${Property.degreeSuffix}")
        null
      case TaskName.affecteds =>
        val vertices = rw.loadGraph().map { case (v, p) => v }
        val affecteds = rw.computeAffecteds(vertices.seq, params.affectedLength)
        fm.saveAffecteds(affecteds)
        null
      case TaskName.rr =>
        //        exp.removeAndRun()
        null
      case TaskName.ar =>
        //        exp.addAndRun()
        null
      case TaskName.s1 =>
        //        exp.streamingUpdates()
        null
      case TaskName.soProbs =>
        rw.loadGraph()
        val (edgeIds, probs) = GraphUtils.computeSecondOrderProbs(params)
        FileManager(params).saveSecondOrderProbs(edgeIds, probs)
      case TaskName.coAuthors =>
        DatasetCleaner.convertJsonFile(params)
      case TaskName.sca =>
        //        exp.streamingCoAuthors()
        StreamingExperiment(params).streamEdges()
        null
      case TaskName.ae => // Empirical analysis of affected vertices, edges, and walks
        //        exp.streamingAffecteds()
        null
      case TaskName.gPairs =>
        println("Reading the walks...")
        val walks = fm.readWalks()
        println("Generating the pairs...")
        val pairs = Word2VecUtils.createPairs(walks, numSkips = params.w2vSkipSize, window =
          params.w2vWindow, params.selfContext)
        println("Extracting the vocabulary...")
        val vocab = Word2VecUtils.createVocabulary(walks)
        println("Writing to the file...")
        fm.saveTargetContextPairs(pairs, vocab, s"w${params.w2vWindow}-s${params.w2vSkipSize}")
        println("Completed!")
    }

    params.cmd match {
      case TaskName.queryPaths =>
      //        val counts = rw.queryPaths(paths)
      //        println(s"Total counts: ${counts.length}")
      //        fm.saveCounts(counts)
      //      case TaskName.probs =>
      //        val probs = rw.computeProbs(paths)
      //        println(s"Total prob entries: ${probs.length}")
      //        fm.save(probs)
      case _ =>
    }
  }
}
