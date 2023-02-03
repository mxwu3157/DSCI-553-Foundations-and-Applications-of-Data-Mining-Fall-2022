name := "hw1"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.1.0"


resolvers ++=Seq(
    "apache-snapshots" at "https://repository.apache.org/snapshots/"
)

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    "org.apache.spark" %% "spark-streaming" % sparkVersion,
    "org.apache.spark" %% "spark-hive" % sparkVersion,
    "org.json4s" %% "json4s-jackson" % "4.0.5"
    
)

