import pyspark 
from pyspark import SparkContext
import json
import sys

import os
import time

if __name__ == '__main__':
    
    #os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-1.8.0-openjdk-amd64'

    
    sc = pyspark.SparkContext(appName = 'task1')
    sc.setLogLevel('WARN')
    
    review_filepath = sys.argv[1]
    business_filepath = sys.argv[2]
    output_filepath_question_a = sys.argv[3]
    output_filepath_question_b = sys.argv[4]
    
    output2={}
   
    #m1
    s1 = time.time()
    
    reviewRDD = sc.textFile(review_filepath) \
        .map(json.loads) \
        .map(lambda x : (x['business_id'], x['stars'])) \
        .cache() 
    businessRDD = sc.textFile(business_filepath) \
        .map(json.loads) \
        .map(lambda x: (x['business_id'], x['city'])) \
        .cache()
        
    
    joinRDD = businessRDD.join(reviewRDD) \
        .map(lambda x : (x[1][0], x[1][1])) \
        .aggregateByKey((0,0), lambda x, y: (x[0] + 1, x[1] + y), lambda x, y: (x[0] + y[0], x[1] + y[1])) \
        .mapValues(lambda x : x[1]/x[0]) \
        .collect()
        
    
    joinRDD.sort( key = lambda x : (-x[1],x[0]))
    print(joinRDD[:10])
    
    e1 = time.time()
    
    
    
    #write to file
    with open(output_filepath_question_a, 'w') as f:
        f.write("city,stars\n")
        for pair in joinRDD:
            f.write("%s,%.1f\n" % (pair[0], pair[1]))
        
    f.close()
    
    print('#####################################')
    
    #m2
    s2 = time.time()
    
    reviewRDD = sc.textFile(review_filepath) \
        .map(json.loads) \
        .map(lambda x : (x['business_id'], x['stars'])) \
        .cache() 
    businessRDD = sc.textFile(business_filepath) \
        .map(json.loads) \
        .map(lambda x: (x['business_id'], x['city'])) \
        .cache()
        
    
    joinRDD = businessRDD.join(reviewRDD) \
        .map(lambda x : (x[1][0], x[1][1])) \
        .aggregateByKey((0,0), lambda x, y: (x[0] + 1, x[1] + y), lambda x, y: (x[0] + y[0], x[1] + y[1])) \
        .mapValues(lambda x : x[1]/x[0]) \
        .sortBy(lambda x: (-x[1],x[0])) \
        .take(10)
        
    
    print(joinRDD)
    e2 = time.time()
    
    
    
    #record time
    output2['m1'] = e1-s1
    
    output2['m2'] = e2-s2
    
    output2['reason'] = 'The Spark sorting cost more time than Python list sorting. Possible reason is that Python sort the list in memory, as long as the list can fit in the memory, it will be fast; while Spark spend most of the time on serialization of data rather than peformaing the sorting operations. Serialization distributed the data across clusters with partitions and shuffing, which are all expensive operations in time. '

    
    with open(output_filepath_question_b, 'w') as f:
        json.dump(output2, f)
   