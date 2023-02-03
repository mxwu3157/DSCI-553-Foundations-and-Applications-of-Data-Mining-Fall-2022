import pyspark 
from pyspark import SparkContext
import json
import sys

import os
import time

if __name__ == '__main__':
    
    #os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-1.8.0-openjdk-amd64'
    
    s = time.time()
    
    sc = pyspark.SparkContext(appName = 'task1')
    sc.setLogLevel('WARN')
    
    input_file_path = sys.argv[1]
    output_filepath = sys.argv[2]
    
    output = {}
    
    
    dataRDD = sc.textFile(input_file_path, minPartitions=20).map(json.loads).cache()
    #output['p_size'] = dataRDD.getNumPartitions()

    # total number of reviews
    output['n_review'] = dataRDD.count()
    
    # total number of reviews 2018
    output['n_review_2018'] = dataRDD.filter(lambda review: review['date'][:4] == '2018').count()
    
    # Number of distinct users who wrote reviews
    output['n_user'] = dataRDD.map(lambda review: (1, review['user_id'])).distinct().count()
    
    #top10 users
    output['top10_user'] = dataRDD.map(lambda review: (review['user_id'],1)).reduceByKey(lambda a,b: a+b).takeOrdered(10, key=lambda x: (-x[1], x[0]))
    
    
    #number of distinct business
    output['n_business'] = dataRDD.map(lambda review: (1,review['business_id'])).distinct().count()
    
     #top10 business
    output['top10_business'] = dataRDD.map(lambda review: (review['business_id'],1)).reduceByKey(lambda a,b: a+b).takeOrdered(10, key=lambda x: (-x[1], x[0]))
    
    
    
    print('#####################################')

    
    with open(output_filepath, 'w') as f:
        json.dump(output, f)

    e = time.time()
    print('Time = ', e-s)