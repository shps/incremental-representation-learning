package au.csiro.data61.randomwalk.common

import org.scalatest.FunSuite

/**
  * Created by Hooman on 2018-03-23.
  */
class ExperimentUtilTest extends FunSuite {

  test("testParseCountFile") {
    val dataset = "/Users/Ganymedian/Desktop/dynamic-rw/affected-experiments/output/ae/fbp20q005/"
    val fName = dataset + "counts-0.txt"
    val config = Params(input=fName, output=dataset)
    ExperimentUtil.parseCountFile(config)
  }

  test("cleanAndSaveLabels") {
    val dataset = "/Users/Ganymedian/Desktop/dynamic-rw/output/blog-catalog/"
    val fName = dataset + "group-edges.txt"
    val config = Params(input=fName, output=dataset, delimiter = ",")
    ExperimentUtil.cleanAndSaveLabels(config)
  }

}
