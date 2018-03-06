package au.csiro.data61.randomwalk.common

object Property extends Enumeration {
//  private val suffix = (System.currentTimeMillis()/1000).toString
  val countsSuffix = Value(s"counts")
  val pathSuffix = Value(s"path")
  val removeAndRunSuffix = Value(s"remove-runs")
  val probSuffix = Value(s"probs")
  val degreeSuffix = Value(s"degrees")
  val affecteds = Value(s"affecteds")
  val stepsToCompute = Value(s"steps-to-compute")
  val walkersToCompute = Value(s"walkers-to-compute")
  val edgeIds = Value(s"edge-id-map")
  val soProbs = Value(s"so-probs")
}
