#!/bin/bash

export PYSPARK_PYTHON=/usr/local/bin/python3.6
export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.6
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64

spark-submit --executor-memory 4G --driver-memory 4G word_count.py $1