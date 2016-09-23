package com.intel.webscaleml.examples.logisticRegression

import com.intel.webscaleml.algorithms.logisticRegression._
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import com.intel.webscaleml.algorithms.logisticRegression.Utils.LRUtils

object LogisticRegressionExample {
  case class Params(
                     input: String = null,
                     numIterations: Int = 5,
                     numPartitions: Int = 10,
                     miniBatchFraction: Double = 1.0,
                     weightDivisor: Double = 1.0,
                     compressInput: Boolean = false,
                     encodeType: String = "KryoEncoder")

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("Logistic Regression") {
      head("Logistic Regression: an example app for logistic regression.")
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("numPartitions")
        .text(s"number of partitions, default: ${defaultParams.numPartitions}")
        .action((x, c) => c.copy(numPartitions = x))
      opt[Double]("miniBatchFraction")
        .text(s"miniBatchFraction, default: ${defaultParams.miniBatchFraction}")
        .action((x, c) => c.copy(miniBatchFraction = x))
      opt[Double]("weightDivisor")
        .text(s"weight divisor, default: ${defaultParams.weightDivisor}")
        .action((x, c) => c.copy(weightDivisor = x))
      opt[Boolean]("compressInput")
        .text(s"compress of input data, default: ${defaultParams.compressInput}")
        .action((x, c) => c.copy(compressInput = x))
      opt[String]("encoderClass")
        .text(s"compress of input data, default: ${defaultParams.encodeType}")
        .action((x, c) => c.copy(encodeType = x))
      arg[String]("<input>")
        .required()
        .text("input paths to labeled examples")
        .action((x, c) => c.copy(input = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class com.intel.webscaleml.algorithms.logisticRegression.LogisticRegression \
          |  WebScaleML-algorithms-1.0.0-SNAPSHOT-jar-with-dependencies.jar \
          |  data/sample_data.txt
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"LogisticRegression with $params")
    val sc = new SparkContext(conf)

    val gradient = new LogisticGradient()
    //  val gradient = new ParallelizedLogisticGradient()
    val updater = new SimpleUpdater()

    val optimizer = new GradientDescent(gradient, updater)
      .setNumIterations(params.numIterations)
      .setMiniBatchFraction(params.miniBatchFraction)

    //    val optimizer = new ParallelizedSGD(gradient, updater)
    //      .setNumIterations(params.numIterations)

    //    val updater = new SquaredL2Updater()
    //    val optimizer = new LBFGS(gradient, updater)
    //      .setNumIterations(params.numIterations)
    //      .setRegParam(0.25)

    LRUtils.setEncoder(params.encodeType)
    var res: (Array[Int], Array[Double]) = null.asInstanceOf[(Array[Int], Array[Double])]
    if(params.compressInput) {
      val data = LRUtils.loadFileAsMatrix(sc, params.input, params.numPartitions)
      res = LogisticRegression.train(data, optimizer)
    } else {
      val data = LRUtils.loadFileAsVector(sc, params.input, params.numPartitions)
      res = LogisticRegression.train2(data, optimizer)
    }

    //    val out2 = new java.io.PrintWriter("/home/data/weights.txt")
    //    val global2hdfsMapping = res._1
    //    val weights = res._2
    //    for(i <- 0 until global2hdfsMapping.length) {
    //      out2.println("feature="+global2hdfsMapping(i)+" value="+weights(i))
    //    }
    //    out2.close()
    sc.stop()
  }
}
