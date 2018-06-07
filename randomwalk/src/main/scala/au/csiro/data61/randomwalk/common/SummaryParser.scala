package au.csiro.data61.randomwalk.common

import java.io.{BufferedWriter, File, FileWriter}
import java.nio.file.{Files, Paths}

import better.files._

import scala.collection.parallel.ParSeq
import scala.io.Source

/**
  * Created by Hooman on 2018-05-17.
  */
object SummaryParser {

  val DIR = "summary" + "1528361738"
  val SUMMARY_DIR = s"/Users/Ganymedian/Desktop/$DIR"
  val OUTPUT_DIR = s"/Users/Ganymedian/Desktop/$DIR/final"
  val SCORE_FILE = "score-summary.csv"
  val RW_WALK_FILE = "rw-walk-summary.csv"
  val RW_TIME_FILE = "rw-time-summary.csv"
  val RW_STEP = "rw-step-summary.csv"
  val RW_MAX_ERROR = "rw-max-error-summary.csv"
  val RW_MEAN_ERROR = "rw-mean-error-summary.csv"
  val EPOCH_TIME = "epoch-summary.csv"

  val SPACE = "\\s+"
  val COMMA = ","


  def computeScoresStats(scores: ParSeq[((String, Int, Int, Int, Int), (Int, Float, Float, Float,
    Float))]): Seq[((String, Int, Int, Int, Int), (Float, Float, Float, Float, Float, Float,
    Float, Float))] = {
    return scores.groupBy(_._1).seq.map { case (key, values) =>
      val cleaned = values.map(_._2)
      val sums = cleaned.foldLeft((0, 0f, 0f, 0f, 0f)) { case (
        (_, a, b, c, d), (run, trainAcc, trainF1, testAcc, testF1)) => (run, trainAcc + a,
        trainF1 + b, testAcc + c, testF1 + d)
      }
      val size = cleaned.size
      val meanTrainAcc = sums._2 / size
      val meanTrainF1 = sums._3 / size
      val meanTestAcc = sums._4 / size
      val meanTestF1 = sums._5 / size
      val stdv = cleaned.foldLeft((0, 0.0, 0.0, 0.0, 0.0)) { case ((_, a, b, c, d), (run,
      trainAcc, trainF1, testAcc, testF1)) => (run, Math.pow(Math.abs(trainAcc - meanTrainAcc),
        2) + a, Math.pow(Math.abs(trainF1 - meanTrainF1), 2) + b, Math.pow(Math.abs(testAcc -
        meanTestAcc), 2) + c, Math.pow(Math.abs(testF1 - meanTestF1), 2) + d)
      }

      val stdTrainAcc = Math.sqrt(stdv._2 / size).toFloat
      val stdTrainF1 = Math.sqrt(stdv._3 / size).toFloat
      val stdTestAcc = Math.sqrt(stdv._4 / size).toFloat
      val stdTestF1 = Math.sqrt(stdv._5 / size).toFloat

      (key, (meanTrainAcc, stdTrainAcc, meanTrainF1, stdTrainF1, meanTestAcc, stdTestAcc,
        meanTestF1, stdTestF1))
    }.toSeq
  }

  def computeEpochStats(epochs: ParSeq[((String, Int, Int, Int, Int), (Int, Double))]): Seq[(
    (String, Int, Int, Int, Int), (Double, Float))] = {
    return epochs.groupBy(_._1).seq.map { case (key, values) =>
      val cleaned = values.map(_._2)
      val sums = cleaned.foldLeft((0, 0.0)) { case ((_, a), (run, time)) => (run, time + a)
      }
      val size = cleaned.size
      val meanTime = sums._2 / size
      val stdv = cleaned.foldLeft((0, 0.0)) { case ((_, a), (run, time)) => (run, Math.pow(Math
        .abs(time - meanTime), 2) + a)
      }

      val stdTime = Math.sqrt(stdv._2 / size).toFloat

      (key, (meanTime, stdTime))
    }.toSeq
  }

  def computeRwStats(rws: ParSeq[((String, Int, Int), (Int, Array[Float]))], nSteps: Int): Seq[(
    (String, Int, Int), (Array[Float], Array[Float]))] = {
    return rws.groupBy(_._1).seq.map { case (key, values) =>
      val cleaned = values.map(_._2)

      val sums = cleaned.foldLeft((0, new Array[Float](nSteps))) { case ((_, arr), (run, steps)) =>
        for (i <- 0 until steps.length) {
          arr(i) += steps(i)
        }
        (run, arr)
      }
      val size = cleaned.size
      val means = sums._2.map { case a => a.toFloat / size.toFloat }
      val stdv = cleaned.foldLeft((0, new Array[Double](nSteps))) { case ((_, arr), (run, steps)) =>
        for (i <- 0 until steps.length) {
          arr(i) += Math.pow(Math.abs(steps(i) - means(i)), 2)
        }

        (run, arr)
      }

      val stds = stdv._2.map { case a => Math.sqrt(a / size).toFloat }

      (key, (means, stds))
    }.toSeq
  }

  def readScoresSummary(file: String): ParSeq[((String, Int, Int, Int, Int), (Int, Float, Float,
    Float,
    Float))] = {
    val lines = Source.fromFile(file).getLines.drop(1).toArray.par

    lines.flatMap { triplet =>
      val parts = triplet.split(COMMA).map(_.trim)
      val method = parts(0)
      val numWalk = parts(1).toInt
      val walkLength = parts(2).toInt
      val run = parts(3).toInt
      val step = parts(4).toInt
      val epoch = parts(5).toInt
      val trainAcc = parts(6).toFloat
      val trainF1 = parts(7).toFloat
      val testAcc = parts(8).toFloat
      val testF1 = parts(9).toFloat

      Seq(((method, numWalk, walkLength, step, epoch), (run, trainAcc, trainF1, testAcc, testF1)))
    }
  }

  def readEpochSummary(file: String): ParSeq[((String, Int, Int, Int, Int), (Int, Double))] = {
    val lines = Source.fromFile(file).getLines.drop(1).toArray.par

    lines.flatMap { triplet =>
      val parts = triplet.split(COMMA).map(_.trim)
      val method = parts(0)
      val numWalk = parts(1).toInt
      val walkLength = parts(2).toInt
      val run = parts(3).toInt
      val step = parts(4).toInt
      val epoch = parts(5).toInt
      val time = parts(6).toDouble

      Seq(((method, numWalk, walkLength, step, epoch), (run, time)))
    }
  }

  def readRandomWalkSummary(file: String): ParSeq[((String, Int, Int), (Int, Array[Float]))] = {
    val lines = Source.fromFile(file).getLines.drop(1).toArray.par // drops the header

    lines.flatMap { triplet =>
      val parts = triplet.split(SPACE).map(_.trim)
      val method = parts(0)
      val numWalk = parts(1).toInt
      val walkLength = parts(2).toInt
      val run = parts(3).toInt
      val steps = parts.slice(4, parts.length).map(_.toFloat)

      Seq(((method, numWalk, walkLength), (run, steps)))
    }
  }

  def saveScores(scoresStats: Seq[((String, Int, Int, Int, Int), (Float, Float, Float, Float, Float,
    Float, Float, Float))]): Unit = {

    OUTPUT_DIR.toFile.createIfNotExists(true)
    val file = new File(s"${OUTPUT_DIR}/scores.csv")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write("method\tnw\twl\tstep\tepoch\tx_axis\ttrain_acc\ttrain_acc_std\ttrain_f1" +
      "\ttrain_f1_std" +
      "\ttest_acc\ttest_acc_std\ttest_f1\ttest_f1_std\n")
    bw.write(scoresStats.map { case (p1, p2) => s"${p1._1}\t${p1._2}\t${p1._3}\t${p1._4}\t${
      p1
        ._5
    }\t${p1._4}-${p1._5 + 1}\t${p2._1}\t${p2._2}\t${p2._3}\t${p2._4}\t${p2._5}\t${p2._6}\t${
      p2
        ._7
    }\t${p2._8}"
    }
      .mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveRw(walksStats: Seq[((String, Int, Int), (Array[Float], Array[Float]))], fName: String):
  Unit = {
    OUTPUT_DIR.toFile.createIfNotExists(true)
    val file = new File(s"${OUTPUT_DIR}/$fName.csv")
    val bw = new BufferedWriter(new FileWriter(file))
    val nSteps = walksStats.head._2._1.length
    val stepNumbers = (0 to nSteps - 1).map(i => s"mean_step_$i\tstd_step_$i").mkString("\t")
    bw.write(s"method\tnw\twl\tstep\tmean\tstd\n")
    bw.write(walksStats.map { case (p1, p2) =>
      (p2._1 zip p2._2).map { case (mean, std) => s"$mean\t$std" }.zipWithIndex.map { case (r, i) =>
        s"${p1._1}\t${p1._2}\t${p1._3}\t$i\t$r"
      }.mkString("\n")
    }.mkString("\n"))
    bw.flush()
    bw.close()
  }

  def saveEpochs(epochStats: Seq[((String, Int, Int, Int, Int), (Double, Float))]) = {
    OUTPUT_DIR.toFile.createIfNotExists(true)
    val file = new File(s"${OUTPUT_DIR}/epoch-times.csv")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write("method\tnw\twl\tstep\tepoch\tx_axis\tmean_time\tstd_time\n")
    bw.write(epochStats.map { case (p1, p2) => s"${p1._1}\t${p1._2}\t${p1._3}\t${p1._4}\t${
      p1
        ._5
    }\t${p1._4}-${p1._5 + 1}\t${p2._1}\t${p2._2}"
    }
      .mkString("\n"))
    bw.flush()
    bw.close()
  }

  def main(args: Array[String]) {
    val scoreFile = s"$SUMMARY_DIR/$SCORE_FILE"
    if (Files.exists(Paths.get(scoreFile))) {
      val scores = readScoresSummary(scoreFile)
      val scoresStats = computeScoresStats(scores).sortBy(r => (r._1._1, r._1._2, r._1._3, r._1._4,
        r._1._5))
      saveScores(scoresStats)
    }

    val epochFile = s"$SUMMARY_DIR/$EPOCH_TIME"
    if (Files.exists(Paths.get(epochFile))) {
      val epochs = readEpochSummary(epochFile)
      val epochStats = computeEpochStats(epochs).sortBy(r => (r._1._1, r._1._2, r._1._3, r._1._4, r
        ._1._5))
      saveEpochs(epochStats)
    }


    val walksFile = s"$SUMMARY_DIR/$RW_WALK_FILE"
    if (Files.exists(Paths.get(walksFile))) {
      val walks = readRandomWalkSummary(walksFile)
      val nSteps = walks(0)._2._2.length
      val stepsFile = s"$SUMMARY_DIR/$RW_STEP"
      val steps = readRandomWalkSummary(stepsFile)
      val timesFile = s"$SUMMARY_DIR/$RW_TIME_FILE"
      val times = readRandomWalkSummary(timesFile)
      val walksStats = computeRwStats(walks, nSteps).sortBy(r => (r._1._1, r._1._2, r._1._3))
      val stepsStats = computeRwStats(steps, nSteps).sortBy(r => (r._1._1, r._1._2, r._1._3))
      val timesStats = computeRwStats(times, nSteps).sortBy(r => (r._1._1, r._1._2, r._1._3))

      saveRw(walksStats, "rw-walks")
      saveRw(stepsStats, "rw-steps")
      saveRw(timesStats, "rw-times")

      val meanErrFile = s"$SUMMARY_DIR/$RW_MEAN_ERROR"
      if (Files.exists(Paths.get(meanErrFile))) {
        val meanErrors = readRandomWalkSummary(meanErrFile)
        val maxErrFile = s"$SUMMARY_DIR/$RW_MAX_ERROR"
        val maxErrors = readRandomWalkSummary(maxErrFile)
        val meanErrorsStats = computeRwStats(meanErrors, nSteps).sortBy(r => (r._1._1, r._1._2, r._1
          ._3))
        val maxErrorsStats = computeRwStats(maxErrors, nSteps).sortBy(r => (r._1._1, r._1._2, r._1
          ._3))
        saveRw(meanErrorsStats, "rw-mean-errors")
        saveRw(maxErrorsStats, "rw-max-errors")
      }
    }
  }


}
