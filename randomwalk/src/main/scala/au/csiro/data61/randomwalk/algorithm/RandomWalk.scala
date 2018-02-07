package au.csiro.data61.randomwalk.algorithm

import au.csiro.data61.randomwalk.common.{Params, Property}
import org.apache.log4j.LogManager
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, SparkContext}

import scala.util.Random

trait RandomWalk extends Serializable {

  protected val context: SparkContext
  protected val config: Params
  lazy val partitioner: HashPartitioner = new HashPartitioner(config.rddPartitions)
  /**
    * Routing table is for guiding Spark-engine to co-locate each random-walker with the
    * correct partition in the same executor.
    */
  var routingTable: RDD[Int] = _
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
    * Initializes the first step of the randomwalk, that is a first-order randomwalk.
    *
    * @param paths
    * @param nextFloat random number generator. This enables to assign any type of random number
    *                  generator that can be used for test purposes as well.
    * @return a tuple including (partition-id, (path, current-vertex-neighbors, completed))
    */
  def initFirstStep(paths: RDD[(Int, Array[Int])], nextFloat: () =>
    Float = Random.nextFloat): RDD[(Int, (Array[Int], Array[(Int, Float)], Boolean))] = {
    paths.mapPartitions({ iter =>
      iter.map { case (pId, path: Array[Int]) =>
        val neighbors = GraphMap.getNeighbors(path.head)
        if (neighbors != null && neighbors.length > 0) {
          val (nextStep, _) = RandomSample(nextFloat).sample(neighbors)
          (pId, (path ++ Array(nextStep), neighbors, false))
        } else {
          // It's a deaend.
          (pId, (path, Array.empty[(Int, Float)], true))
        }
      }
    }, preservesPartitioning = true
    )
  }

  /**
    * Uses routing table to partition walkers RDD based on their key (partition-id) and locates
    * specific partition ids to specific executors.
    *
    * @param routingTable
    * @param walkers
    * @return
    */
  def transferWalkersToTheirPartitions(routingTable: RDD[Int], walkers: RDD[(Int, (Array[Int],
    Array[(Int, Float)], Boolean))]): RDD[(Int, (Array[Int], Array[(Int, Float)], Boolean))] = {
    routingTable.zipPartitions(walkers.partitionBy(partitioner)) {
      (_, iter2) =>
        iter2
    }
  }

  /**
    * Extracts the unfinished random-walkers.
    *
    * @param walkers
    * @return
    */
  def filterUnfinishedWalkers(walkers: RDD[(Int, (Array[Int], Array[(Int, Float)], Boolean))])
  : RDD[(Int, (Array[Int], Array[(Int, Float)], Boolean))] = {
    walkers.filter(!_._2._3)
  }

  /**
    * Extracts completed paths.
    *
    * @param walkers
    * @return completed paths.
    */
  def filterCompletedPaths(walkers: RDD[(Int, (Array[Int], Array[(Int, Float)], Boolean))])
  : RDD[Array[Int]] = {
    walkers.filter(_._2._3).map { case (_, (paths, _, _)) =>
      paths
    }
  }

  /**
    * Updates the partition-id (the key) for transferring the walker to its right destination.
    *
    * @param walkers
    * @return
    */
  def prepareWalkersToTransfer(walkers: RDD[(Int, (Array[Int], Array[(Int, Float)], Boolean))])
  : RDD[(Int, (Array[Int], Array[(Int, Float)], Boolean))]

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
