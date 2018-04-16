package au.csiro.data61.randomwalk.algorithm

import org.scalatest.FunSuite

import scala.collection.mutable

class RandomSampleTest extends FunSuite {

  // TODO assert can move to a function for DRY purpose.
  test("Test random sample function") {
    var rValue = 0.1f
    var random = RandomSample(nextFloat = () => rValue)
    assert(random.nextFloat() == rValue)
    val e1 = (1, 1.0f)
    val e2 = (2, 1.0f)
    val e3 = (3, 1.0f)
    val edges = mutable.Set(e1, e2, e3)
    var expected = Seq.empty[(Int, Float)]
    for (e <- edges) {
      expected ++= Seq(e)
    }
    assert(random.sample(edges) == expected(0))
    rValue = 0.4f
    random = RandomSample(nextFloat = () => rValue)
    assert(random.sample(edges) == expected(1))
    rValue = 0.7f
    random = RandomSample(nextFloat = () => rValue)
    assert(random.sample(edges) == expected(2))
  }
}
