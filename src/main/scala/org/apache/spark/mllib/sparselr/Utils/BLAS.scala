package org.apache.spark.mllib.sparselr.Utils

import org.apache.spark.internal.Logging

object BLAS extends Serializable with Logging{
  def dot(a: Double, y: Vector): Unit = {
    y match {
      case hy: HashedSparseVector =>
        dot(a, hy)
      case _ =>
        throw new IllegalArgumentException(s"dot doesn't support (Double, ${y.getClass}).")
    }
  }

  private def dot(a: Double, y: HashedSparseVector): Unit = {
    y.iterator.foreach {keyVal =>
      y(keyVal._1) = a * keyVal._2
    }
  }

  /**
   * dot(x, y)
   */
  def dot(x: Vector, y: Vector): Double = {
    (x, y) match {
      case (_: CompressedSparseVector, _: DenseVector) | (_: SparseVector, _: DenseVector) =>
        _dot(x, y)
      case _ =>
        throw new IllegalArgumentException(s"dot doesn't support (${x.getClass}, ${y.getClass}).")
    }
  }

  private def _dot(x: Vector, y: Vector): Double = {
    var margin = 0.0
    x.iterator.foreach { keyVal =>
      margin += (y(keyVal._1) * keyVal._2)
    }
    return margin
  }

  /**
   * res = x * y|z
   */
  def dot(x: Vector, y: Vector, z: Vector): Double = {
    (x, y, z) match {
      case (cx: CompressedSparseVector, dy: DenseVector, hz: HashedSparseVector) =>
        dot(cx, dy, hz)
      case _ =>
        throw new IllegalArgumentException(s"dot doesn't support (${x.getClass}, ${y.getClass}).")
    }
  }

  private def dot(x: CompressedSparseVector, y: DenseVector, z: HashedSparseVector): Double = {
    var weightSum = 0.0
    x.iterator.foreach { keyVal =>
      if(!z.contains(keyVal._1)) {
        z(keyVal._1) = y(keyVal._1)
      }
      weightSum += (z(keyVal._1) * keyVal._2)
    }
    weightSum
  }

  /**
   * y += a * x
   */
    def axpy(a: Double, x: Vector, y: Vector): Unit = {
    (x, y) match {
        case (_: CompressedSparseVector, _: HashedSparseVector)
             | (_: HashedSparseVector, _: DenseVector)
             | (_: SparseVector, _: HashedSparseVector) =>
          _axpy(a, x, y)
        case _ =>
          throw new IllegalArgumentException(s"dot doesn't support (${x.getClass}, ${y.getClass}).")
      }
    }

  /**
   * y += a * x
   */
  private def _axpy(a: Double, x: Vector, y: Vector): Unit = {
    x.iterator.foreach { keyVal =>
      y(keyVal._1) = a * keyVal._2 + y(keyVal._1)
    }
  }
}
