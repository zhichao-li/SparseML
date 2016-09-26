package org.apache.spark.mllib.sparselr

import scala.collection.mutable
import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}
import org.apache.spark.SparkEnv
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.sparselr.Utils._

/**
* :: DeveloperApi ::
* Class used to solve an optimization problem using Limited-memory BFGS.
* Reference: [[http://en.wikipedia.org/wiki/Limited-memory_BFGS]]
* @param gradient Gradient function to be used.
* @param updater Updater to be used to update weights after every iteration.
*/
@DeveloperApi
class LBFGS(private var gradient: Gradient, private var updater: Updater)
  extends Optimizer with Logging {

  private var numCorrections = 10
  private var convergenceTol = 1E-4
  private var maxNumIterations = 100
  private var regParam = 0.0

  /**
   * Set the number of corrections used in the LBFGS update. Default 10.
   * Values of numCorrections less than 3 are not recommended; large values
   * of numCorrections will result in excessive computing time.
   * 3 < numCorrections < 10 is recommended.
   * Restriction: numCorrections > 0
   */
  def setNumCorrections(corrections: Int): this.type = {
    assert(corrections > 0)
    this.numCorrections = corrections
    this
  }

  /**
   * Set the convergence tolerance of iterations for L-BFGS. Default 1E-4.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * This value must be nonnegative. Lower convergence values are less tolerant
   * and therefore generally cause more iterations to be run.
   */
  def setConvergenceTol(tolerance: Double): this.type = {
    this.convergenceTol = tolerance
    this
  }

  /**
   * Set the maximal number of iterations for L-BFGS. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.maxNumIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the gradient function (of the loss function of one single data example)
   * to be used for L-BFGS.
   */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }

  /**
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  override def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = LBFGS.runLBFGS(
      data,
      gradient,
      updater,
      numCorrections,
      convergenceTol,
      maxNumIterations,
      regParam,
      initialWeights)
    weights
  }
}

/**
* :: DeveloperApi ::
* Top-level method to run L-BFGS.
*/
@DeveloperApi
object LBFGS extends Logging {
  /**
   * Run Limited-memory BFGS (L-BFGS) in parallel.
   * Averaging the subgradients over different partitions is performed using one standard
   * spark map-reduce in each iteration.
   *
   * @param data - Input data for L-BFGS. RDD of the set of data examples, each of
   *               the form (label, [feature values]).
   * @param gradient - Gradient object (used to compute the gradient of the loss function of
   *                   one single data example)
   * @param updater - Updater function to actually perform a gradient step in a given direction.
   * @param numCorrections - The number of corrections used in the L-BFGS update.
   * @param convergenceTol - The convergence tolerance of iterations for L-BFGS which is must be
   *                         nonnegative. Lower values are less tolerant and therefore generally
   *                         cause more iterations to be run.
   * @param maxNumIterations - Maximal number of iterations that L-BFGS can be run.
   * @param regParam - Regularization parameter
   *
   * @return A tuple containing two elements. The first element is a column matrix containing
   *         weights for every feature, and the second element is an array containing the loss
   *         computed for every iteration.
   */
  def runLBFGS(
                data: RDD[(Double, Vector)],
                gradient: Gradient,
                updater: Updater,
                numCorrections: Int,
                convergenceTol: Double,
                maxNumIterations: Int,
                regParam: Double,
                initialWeights: Vector): (Vector, Array[Double]) = {

    val lossHistory = mutable.ArrayBuilder.make[Double]

    val numExamples = data.count()

    val costFun =
      new CostFun(data, gradient, updater, regParam, numExamples)

    val lbfgs = new BreezeLBFGS[BDV[Double]](maxNumIterations, numCorrections, convergenceTol)

    val states =
      lbfgs.iterations(new CachedDiffFunction(costFun), new BDV[Double](initialWeights.toArray).toDenseVector)

    /**
     * NOTE: lossSum and loss is computed using the weights from the previous iteration
     * and regVal is the regularization value computed in the previous iteration as well.
     */
    var state = states.next()
    while (states.hasNext) {
      lossHistory += state.value
      state = states.next()
    }
    lossHistory += state.value
    val weights = new DenseVector(state.x.data)

    val lossHistoryArray = lossHistory.result()

    logInfo("LBFGS.runLBFGS finished. Last 10 losses %s".format(
      lossHistoryArray.takeRight(10).mkString(", ")))

    (weights, lossHistoryArray)
  }

  /**
   * CostFun implements Breeze's DiffFunction[T], which returns the loss and gradient
   * at a particular point (weights). It's used in Breeze's convex optimization routines.
   */
  private class CostFun(
                         data: RDD[(Double, Vector)],
                         gradient: Gradient,
                         updater: Updater,
                         regParam: Double,
                         numExamples: Long) extends DiffFunction[BDV[Double]] {

    var hashmap: HashedSparseVector = null

    override def calculate(weights: BDV[Double]): (Double, BDV[Double]) = {
      val w = Vectors.dense(weights.data)
      val n = w.toArray.length
      val bcW = data.context.broadcast(w)

      val localGradient = gradient

      val (gradientSum, lossSum) = data.mapPartitions { points =>
        var loss = 0.0
        val gradientPerPartition = new HashedSparseVector

        points.foreach { point =>
          loss += localGradient.compute(point._2, point. _1, bcW.value, gradientPerPartition)
        }
        Iterator((gradientPerPartition, loss))
      }.reduce { (first, second) =>
          val iterSecond = second._1.hashmap.int2DoubleEntrySet.fastIterator()
          while (iterSecond.hasNext()) {
            val entry = iterSecond.next()
            first._1.hashmap.addTo(entry.getIntKey, entry.getDoubleValue)
          }
          (first._1, first._2 + second._2)
        }
      hashmap = if(hashmap == null) new HashedSparseVector() else hashmap
      /**
       * regVal is sum of weight squares if it's L2 updater;
       * for other updater, the same logic is followed.
       */
      val regVal = updater.compute(w, hashmap, 0, 1, regParam)._2

      val loss = lossSum / numExamples + regVal
      /**
       * It will return the gradient part of regularization using updater.
       *
       * Given the input parameters, the updater basically does the following,
       *
       * w' = w - thisIterStepSize * (gradient + regGradient(w))
       * Note that regGradient is function of w
       *
       * If we set gradient = 0, thisIterStepSize = 1, then
       *
       * regGradient(w) = w - w'
       *
       * TODO: We need to clean it up by separating the logic of regularization out
       *       from updater to regularizer.
       */
      // The following gradientTotal is actually the regularization part of gradient.
      // Will add the gradientSum computed from the data with weights in the next step.
      val gradientTotal = w.copy

      val update = updater.compute(w, hashmap, 1, 1, regParam)

      (0 until n).foreach { i =>
        gradientTotal(i) -= update._1(i)
        gradientTotal(i) += gradientSum.hashmap.get(i) / numExamples
      }

      SparkEnv.get.blockManager.removeBroadcast(bcW.id, true)

      (loss, new BDV[Double](gradientTotal.toArray))
    }
  }
}
