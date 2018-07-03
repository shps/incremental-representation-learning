package au.csiro.data61.randomwalk.common

object Property extends Enumeration {
//  private val suffix = (System.currentTimeMillis()/1000).toString
  val countsSuffix = Value("counts")
  val pathSuffix = Value("path")
  val afsSuffix = Value("afs")
  val removeAndRunSuffix = Value("remove-runs")
  val probSuffix = Value("probs")
  val degreeSuffix = Value("degrees")
  val graphStatsSuffix = Value("graphstats")
  val affecteds = Value("affecteds")
  val stepsToCompute = Value("steps-to-compute")
  val walkersToCompute = Value("walkers-to-compute")
  val timesToCompute = Value("time-to-compute")
  val meanErrors = Value("mean-errors")
  val maxErrors = Value("max-errors")
  val edgeIds = Value("edge-id-map")
  val soProbs = Value("so-probs")
}
