package au.csiro.data61.randomwalk.common

import au.csiro.data61.randomwalk.common.CommandParser.RrType.RrType
import au.csiro.data61.randomwalk.common.CommandParser.{RrType, TaskName, WalkType}
import au.csiro.data61.randomwalk.common.CommandParser.TaskName.TaskName
import au.csiro.data61.randomwalk.common.CommandParser.WalkType.WalkType


case class Params(walkLength: Int = 80,
                  numWalks: Int = 10,
                  weighted: Boolean = true,
                  directed: Boolean = false,
                  p: Float = 1.0f,
                  q: Float = 1.0f,
                  input: String = null,
                  output: String = null,
                  rddPartitions: Int = 200,
                  partitioned: Boolean = false,
                  affectedLength: Int = 3,
                  numRuns: Int = 1,
                  nodes: String = "",
                  rrType: RrType = RrType.m1,
                  numVertices: Int = 34,
                  seed: Long = System.currentTimeMillis(),
                  wType: WalkType = WalkType.firstorder,
                  delimiter:String = "\\s+",
                  cmd: TaskName = TaskName.ar) extends AbstractParams[Params]