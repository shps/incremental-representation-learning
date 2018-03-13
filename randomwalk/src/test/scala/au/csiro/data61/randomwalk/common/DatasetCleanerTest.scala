package au.csiro.data61.randomwalk.common

import org.scalatest.FunSuite


/**
  * Created by Hooman on 2018-02-27.
  */
class DatasetCleanerTest extends FunSuite {

  private val dataset = "/Users/Ganymedian/Desktop/dynamic-rw/datasets/"

  test("testCheckDataSet") {
    val fName = dataset + "soc-wiki-vote.txt"
    val initId = 1
    val config = Params(input = fName)
    DatasetCleaner.checkDataSet(config, initId)

  }

  case class CoAuthor(a1: String, a2: String, year: Int)

  test("jsonConvertor") {
    val fName = dataset + "test.json"
    val output = dataset
    val config = Params(input = fName, output = output)
    DatasetCleaner.convertJsonFile(config)
  }

}
