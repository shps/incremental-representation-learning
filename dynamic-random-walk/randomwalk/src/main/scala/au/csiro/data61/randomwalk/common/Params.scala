package au.csiro.data61.randomwalk.common

import au.csiro.data61.randomwalk.common.CommandParser.RrType.RrType
import au.csiro.data61.randomwalk.common.CommandParser.TaskName.TaskName
import au.csiro.data61.randomwalk.common.CommandParser.{RrType, TaskName}


case class Params(walkLength: Int = 80,
                  numWalks: Int = 10,
                  weighted: Boolean = false,
                  directed: Boolean = false,
                  p: Float = 1.0f,
                  q: Float = 1.0f,
                  input: String = null,
                  output: String = null,
                  numRuns: Int = 1,
                  rrType: RrType = RrType.m1,
                  w2vWindow: Int = 1,
                  w2vSkipSize: Int = 2,
                  savePeriod: Int = 100,
                  initEdgeSize: Int = 0,
                  edgeStreamSize: Int = 100,
                  maxSteps: Int = 10,
                  logErrors: Boolean = false,
                  seed: Long = 1234,
                  selfContext: Boolean = false,
                  forceSkipSize: Boolean = false,
                  allWalks: Boolean = true,
                  countSccs: Boolean = false,
                  fixedGraph: Boolean = false,
                  O: Float = 0.5f,
                  grouped: Boolean = false,
                  delimiter: String = "\\s+",
                  cmd: TaskName = TaskName.sca) extends AbstractParams[Params]