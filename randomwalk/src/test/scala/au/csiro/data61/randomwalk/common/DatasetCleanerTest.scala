package au.csiro.data61.randomwalk.common

import org.scalatest.FunSuite

/**
  * Created by Hooman on 2018-02-27.
  */
class DatasetCleanerTest extends FunSuite {

  private val dataset = "/Users/Ganymedian/Desktop/dynamic-rw/datasets/"

  test("testCheckDataSet") {
    val fName = dataset + "facebook_combined.txt"
    val initId = 0
    val config = Params(input = fName)
    DatasetCleaner.checkDataSet(config, initId)

  }

}
