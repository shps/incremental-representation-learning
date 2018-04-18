package au.csiro.data61.randomwalk.common

import org.scalatest.FunSuite

import scala.collection.parallel.ParSeq

/**
  * Created by Hooman on 2018-04-09.
  */
class Word2VecUtilsTest extends FunSuite {

  test("testCreatePairs") {

    val walks = ParSeq(Seq(1, 2, 1, 3, 4, 5), Seq(6,7,8,9,10))
    var pairs = Word2VecUtils.createPairs(walks, numSkips = 2, window = 2, selfContext = false)
    print(pairs.map{case (t, c) => s"($t , $c)"}.mkString("\n"))
    pairs = Word2VecUtils.createPairs(walks, numSkips = 2, window = 2, selfContext = true)
    print(pairs.map{case (t, c) => s"($t , $c)"}.mkString("\n"))
  }

}
