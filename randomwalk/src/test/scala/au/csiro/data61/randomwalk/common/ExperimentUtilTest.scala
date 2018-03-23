package au.csiro.data61.randomwalk.common

import org.scalatest.FunSuite

/**
  * Created by Hooman on 2018-03-23.
  */
class ExperimentUtilTest extends FunSuite {

  private val dataset = "/Users/Ganymedian/Desktop/dynamic-rw/affected-experiments/output/ae/fbp20q005/"

  test("testParseCountFile") {
    val fName = dataset + "counts-0.txt"
    val config = Params(input=fName, output=dataset)
    ExperimentUtil.parseCountFile(config)
  }

}
