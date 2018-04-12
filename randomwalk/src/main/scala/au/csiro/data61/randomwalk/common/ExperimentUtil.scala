package au.csiro.data61.randomwalk.common

/**
  * Created by Hooman on 2018-03-23.
  */
object ExperimentUtil {


  def parseCountFile(config: Params): Unit = {
    val fm = FileManager(config)
    val counts = fm.readCountFile()
    val groups = counts.groupBy(_._2).map { case (c, occurs) =>
      (c, occurs.size)
    }.toSeq
    val max = groups.maxBy(_._1)._1
    val occurs = new Array[Int](max + 1)
    for ((c, o) <- groups) {
      occurs(c) = o
    }
    fm.saveCountGroups(occurs)
  }

  def cleanAndSaveLabels(config: Params): Unit = {
    val fm = FileManager(config)
    val lFile = fm.readCountFile()
    val vLabels = lFile.groupBy(_._1).map { case (v, labels) =>
      (v, labels.map(_._2))
    }.toSeq.seq.sortWith(_._1 < _._1).par
    fm.saveLabels(vLabels)
  }
}
