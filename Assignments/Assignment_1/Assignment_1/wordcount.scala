 

import org.apache.spark.rdd.RDD
import scala.io.Codec.string2codec
import scala.io.Source
import scala.reflect.io.File
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object WordCount {
    def main(args: Array[String]) {
  
        /* configure spark application */
        val conf = new SparkConf().setAppName("Spark Scala WordCount Example").setMaster("local[1]")
  
        /* spark context*/
        val sc = new SparkContext(conf)
  
        /* map */
        var map = sc.textFile("text.txt").flatMap(line => line.split(" ")).map(word => (word,1))
  
        /* reduce */
        var counts = map.reduceByKey(_ + _)
  
        /* print */
        counts.collect().foreach(println)
  
        /* or save the output to file */
        counts.saveAsTextFile("result.txt")
  
        sc.stop()
    }
}


