package au.csiro.data61.randomwalk.common

object Property extends Enumeration {
  val afsSuffix = Value("afs")
  val biggestScc = Value("bscc")
  val degreeSuffix = Value("degrees")
  val graphStatsSuffix = Value("graphstats")
  val stepsToCompute = Value("steps-to-compute")
  val walkersToCompute = Value("walkers-to-compute")
  val timesToCompute = Value("time-to-compute")
  val meanErrors = Value("mean-errors")
  val maxErrors = Value("max-errors")
}
