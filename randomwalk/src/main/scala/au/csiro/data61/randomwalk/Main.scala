package au.csiro.data61.randomwalk

import au.csiro.data61.randomwalk.algorithm.{Experiments, UniformRandomWalk}
import au.csiro.data61.randomwalk.common.CommandParser.TaskName
import au.csiro.data61.randomwalk.common.{CommandParser, FileManager, GraphUtils, Params}
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
    val exp = Experiments(params)
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
        fm.saveDegrees(rw.degrees())
        null
      case TaskName.affecteds =>
        val vertices = rw.loadGraph().map { case (v, p) => v }
        val affecteds = rw.computeAffecteds(vertices, params.affectedLength)
        fm.saveAffecteds(affecteds)
        null
      case TaskName.rr =>
        exp.removeAndRun()
        null
      case TaskName.ar =>
        exp.addAndRun()
        null
      case TaskName.s1 =>
        exp.streamingUpdates()
        null
      case TaskName.soProbs =>
        rw.loadGraph()
        val (edgeIds, probs) = GraphUtils.computeSecondOrderProbs(params)
        FileManager(params).saveSecondOrderProbs(edgeIds, probs)
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
