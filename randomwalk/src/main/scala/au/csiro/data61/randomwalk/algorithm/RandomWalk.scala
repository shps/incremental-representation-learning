package au.csiro.data61.randomwalk.algorithm

import au.csiro.data61.randomwalk.common.{Params, Property}
import org.apache.log4j.LogManager
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, SparkContext}

trait RandomWalk extends Serializable {

  protected val context: SparkContext
  protected val config: Params
  lazy val partitioner: HashPartitioner = new HashPartitioner(config.rddPartitions)
  /**
    * Routing table is for guiding Spark-engine to co-locate each random-walker with the
    * correct partition in the same executor.
    */
  lazy val logger = LogManager.getLogger("rwLogger")
  var nVertices: Int = 0
  var nEdges: Int = 0

  /**
    * Loads a graph from an edge list.
    *
    * @return an RDD of (srcId, Array(srcId)). Array(srcId) in fact contains the first step of
    *         the randomwalk that is the source vertex itself.
    */
  def loadGraph(): RDD[(Int, Array[Int])]

  /**
    * Writes the given paths to the disk.
    *
    * @param paths
    * @param partitions
    * @param output
    */
  def save(paths: RDD[Array[Int]], partitions: Int, output: String) = {

    paths.map {
      case (path) =>
        val pathString = path.mkString("\t")
        s"$pathString"
    }.repartition(partitions).saveAsTextFile(s"${output}/${Property.pathSuffix}")
  }
}
