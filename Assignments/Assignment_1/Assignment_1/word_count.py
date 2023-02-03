from pyspark import SparkContext

import os
import sys


sc = SparkContext(appName='wordCount')

input_file_path = sys.argv[1]
textRDD = sc.textFile(input_file_path)


counts = textRDD.flatMap(lambda line: line.split(' ')) \
    .map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b).collect()

with open('result.txt', 'w') as f:
    for each_word in counts:
        print(each_word)
        f.write(str(each_word)+'\n')

f.close()
