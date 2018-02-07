package au.csiro.data61.randomwalk.common

import au.csiro.data61.randomwalk.common.CommandParser.RrType.RrType
import au.csiro.data61.randomwalk.common.CommandParser.{RrType, TaskName}
import au.csiro.data61.randomwalk.common.CommandParser.TaskName.TaskName


case class Params(walkLength: Int = 80,
                  numWalks: Int = 10,
                  weighted: Boolean = true,
                  directed: Boolean = false,
                  input: String = null,
                  output: String = null,
                  rddPartitions: Int = 200,
                  partitioned: Boolean = false,
                  affectedLength: Int = 3,
                  numRuns: Int = 1,
                  nodes: String = "",
                  rrType: RrType = RrType.m1,
                  cmd: TaskName = TaskName.firstorder) extends AbstractParams[Params]