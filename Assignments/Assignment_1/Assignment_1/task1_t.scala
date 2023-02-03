import org.apache.spark.rdd.RDD
import scala.io.Codec.string2codec
import scala.io.Source
import scala.reflect.io.File
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
//import org.json4s.native.Json
//import org.json4s.DefaultFormats
import org.json4s._
//import org.json4s.native.JsonMethods._
import org.json4s.jackson.JsonMethods._
import com.fasterxml.jackson.databind.SerializationFeature
//org.json4s.jackson.JsonMethods.mapper.configure(SerializationFeature.CLOSE_CLOSEABLE, false)


object task1 {
    def main(args: Array[String]) {

        val input_file_path = args(0)
        val output_filepath = args(1)
        
        val conf = new SparkConf().setAppName("Spark Scala WordCount Example").setMaster("local[1]")
        val sc = new SparkContext(conf)
        sc.setLogLevel("ERROR")
        
        implicit val formats: Formats = DefaultFormats
        case class Review(review_id: String, user_id: String, business_id: String, stars: String, useful: String, funny: String, cool: String, text: String, date: String)
        //implicit val formats: DefaultFormats.type = DefaultFormats


        val dataRDD = sc.textFile(input_file_path).map(parse)).cache()
        println("##########")
        dataRDD.foreach(println)

        //.map(line=>parse(line)).cache()
       
        //output += ("n_reviews" -> dataRDD.count)
        val A_ans = dataRDD.count()
        //val B_ans = dataRDD.filter(review=> review.date.contains("2018"))
  
        val output = Map(
            "n_reviews"->A_ans
            //"n_review_2018"->B_ans
        )
        
        println(A_ans)
        //Json(DefaultFormats).write(output)
        sc.stop()
    }
}


