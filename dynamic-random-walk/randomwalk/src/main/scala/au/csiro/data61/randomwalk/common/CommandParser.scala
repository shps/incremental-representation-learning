package au.csiro.data61.randomwalk.common

import scopt.OptionParser

object CommandParser {

  object TaskName extends Enumeration {
    type TaskName = Value
    val sca, gPairs = Value
  }

  object RrType extends Enumeration {
    type RrType = Value
    val m1, m2, m3, m4 = Value
  }

  val WALK_LENGTH = "walkLength"
  val NUM_WALKS = "numWalks"
  val WEIGHTED = "weighted"
  val DIRECTED = "directed"
  val NUM_RUNS = "nRuns"
  val RR_TYPE = "rrType"
  val P = "p"
  val Q = "q"
  val INPUT = "input"
  val OUTPUT = "output"
  val CMD = "cmd"
  val SEED = "seed"
  val DELIMITER = "d"
  val W2V_WINDOW = "w2vWindow"
  val W2V_SKIP_SIZE = "w2vSkip"
  val SAVE_PERIOD = "save"
  val LOG_ERRORS = "logErrors"
  val INIT_EDGE_Size = "initEdgeSize"
  val EDGE_STREAM_Size = "edgeStreamSize"
  val MAX_STEPS = "maxSteps"
  val SELF_CONTEXT = "selfContext"
  val FORCE_SKIP_SIZE = "forceSkipSize"
  val ALL_WALKS = "allWalks"
  val COUNT_NUM_SCC = "countScc"
  val FIXED_GRAPH = "fixedGraph"
  val O= "o"
  val GROUPED = "grouped"

  private lazy val defaultParams = Params()
  private lazy val parser = new OptionParser[Params]("2nd Order Random Walk + Word2Vec") {
    head("Main")
    opt[Int](WALK_LENGTH)
      .text(s"walkLength: ${defaultParams.walkLength}")
      .action((x, c) => c.copy(walkLength = x))
    opt[Int](NUM_WALKS)
      .text(s"numWalks: ${defaultParams.numWalks}")
      .action((x, c) => c.copy(numWalks = x))
    opt[Int](NUM_RUNS)
      .text(s"numWalks: ${defaultParams.numRuns}")
      .action((x, c) => c.copy(numRuns = x))
    opt[Double](P)
      .text(s"numWalks: ${defaultParams.p}")
      .action((x, c) => c.copy(p = x.toFloat))
    opt[Double](Q)
      .text(s"numWalks: ${defaultParams.q}")
      .action((x, c) => c.copy(q = x.toFloat))
    opt[Int](INIT_EDGE_Size)
      .text(s"Percentage of edges to be used to construct the initial graph before streaming: ${defaultParams.initEdgeSize}")
      .action((x, c) => c.copy(initEdgeSize = x))
    opt[Int](EDGE_STREAM_Size)
      .text(s"Percentage of edges to stream at every step: ${defaultParams.edgeStreamSize}")
      .action((x, c) => c.copy(edgeStreamSize = x))
    opt[Int](SAVE_PERIOD)
      .text(s"Save Period: ${defaultParams.savePeriod}")
      .action((x, c) => c.copy(savePeriod = x))
    opt[Int](MAX_STEPS)
      .text(s"Max number of steps to run experiments: ${defaultParams.maxSteps}")
      .action((x, c) => c.copy(maxSteps = x))
    opt[Int](W2V_WINDOW)
      .text(s"Word2Vec window size: ${defaultParams.w2vWindow}")
      .action((x, c) => c.copy(w2vWindow = x))
    opt[Int](W2V_SKIP_SIZE)
      .text(s"Word2Vec skip size: ${defaultParams.w2vSkipSize}")
      .action((x, c) => c.copy(w2vSkipSize = x))
    opt[Long](SEED)
      .text(s"seed: ${defaultParams.seed}")
      .action((x, c) => c.copy(seed = x))
    opt[Boolean](WEIGHTED)
      .text(s"weighted: ${defaultParams.weighted}")
      .action((x, c) => c.copy(weighted = x))
    opt[Boolean](DIRECTED)
      .text(s"directed: ${defaultParams.directed}")
      .action((x, c) => c.copy(directed = x))
    opt[Boolean](LOG_ERRORS)
      .text(s"Log Errors (increases run time): ${defaultParams.logErrors}")
      .action((x, c) => c.copy(logErrors = x))
    opt[Boolean](SELF_CONTEXT)
      .text(s"Accept target-context pairs where target=context: ${defaultParams.selfContext}")
      .action((x, c) => c.copy(selfContext = x))
    opt[Boolean](FIXED_GRAPH)
      .text(s"Use the same graph among different runs: ${defaultParams.fixedGraph}")
      .action((x, c) => c.copy(fixedGraph = x))
    opt[Boolean](FORCE_SKIP_SIZE)
      .text(s"Force to generate pairs equal to skipSize: ${defaultParams.forceSkipSize}")
      .action((x, c) => c.copy(forceSkipSize = x))
    opt[Boolean](COUNT_NUM_SCC)
      .text(s"Count number of strongly connected components: ${defaultParams.countSccs}")
      .action((x, c) => c.copy(countSccs = x))
    opt[Boolean](ALL_WALKS)
      .text(s"Include all walks to generate sample pairs: ${defaultParams.allWalks}")
      .action((x, c) => c.copy(allWalks = x))
    opt[Boolean](GROUPED)
      .text(s"Are edges already grouped into steps for streaming: ${defaultParams.grouped}")
      .action((x, c) => c.copy(grouped = x))
    opt[Double](O)
      .text(s"Percentage of new walks to draw from old walks: ${defaultParams.O}")
      .action((x, c) => c.copy(O = x.toFloat))
    opt[String](INPUT)
      .required()
      .text("Input edge file path: empty")
      .action((x, c) => c.copy(input = x))
    opt[String](DELIMITER)
      .text("Delimiter: ")
      .action((x, c) => c.copy(delimiter = x))
    opt[String](OUTPUT)
      .required()
      .text("Output path: empty")
      .action((x, c) => c.copy(output = x))
    opt[String](CMD)
      .required()
      .text(s"command: ${defaultParams.cmd.toString}")
      .action((x, c) => c.copy(cmd = TaskName.withName(x)))
    opt[String](RR_TYPE)
      .text(s"RR Type: ${defaultParams.rrType.toString}")
      .action((x, c) => c.copy(rrType = RrType.withName(x)))
  }

  def parse(args: Array[String]) = {
    parser.parse(args, defaultParams)
  }
}
