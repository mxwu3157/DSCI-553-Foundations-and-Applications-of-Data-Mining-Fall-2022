from pyspark import SparkContext
import sys,os
import math
import time
from itertools import combinations
import random
from graphframes import *
from functools import reduce
from pyspark.sql.functions import col, lit, when
from pyspark.sql import SparkSession

t0 = time.time()


input_file = 'test_data.csv'
#input_file = '../resource/asnlib/publicdata/ub_sample_data.csv'
output_file = 'result1.txt'

threshold = 2

#threshold = int(sys.argv[1])
#input_file = sys.argv[2]
#output_file = sys.argv[3]


sc = SparkContext('local[*]', 'task1')
sc.setLogLevel('WARN')
spark = SparkSession(sc)
lines = sc.textFile(input_file)
header = lines.first()
lines = lines.filter(lambda x: x != header)

lines = lines.map(lambda x: x.split(','))

user_reviews = lines.groupByKey().mapValues(set)

vertices = lines.map(lambda x: (x[0])).distinct().map(lambda x: (x,))

edges = user_reviews.cartesian(user_reviews) \
    .filter(lambda x: x[0]!=x[1]) \
    .map(lambda x: (x[0][0], x[1][0], len(x[0][1].intersection(x[1][1])))) \
    .filter(lambda x: x[2]>=threshold).map(lambda x : (x[0],x[1]))

    
src_nodes = edges.map(lambda x: x[0])
end_nodes = edges.map(lambda x: x[1])
vertices = src_nodes.union(end_nodes).distinct().map(lambda x: (x,))

for x in vertices.collect():
    print(x)








