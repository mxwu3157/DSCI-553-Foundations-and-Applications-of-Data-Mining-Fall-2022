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
   
    # main body
    t1 = time.time()
    
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
        .cache()

        
    t2 = time.time()   
        
    #m1
    m1Data = joinRDD.collect()
    m1Data.sort( key = lambda x : (-x[1],x[0]))
    print(m1Data[:10])
    
    t3 = time.time()
    
    print('#####################################')
    
    #m2
    m2Data = joinRDD.sortBy(lambda x: (-x[1],x[0])).take(10)
    print(m2Data)
    
    t4 = time.time()    
    
    
    
    
     #record time
    output2['m1'] = t3-t1
    
    output2['m2'] = (t4-t1) - (t3-t2)
    
    output2['reason'] = 'Spark sorting is more efficient than python sorting in this case. Possible reason can be that Spark utilized the distributed structure by sorting the partitions in parallel in different clusters then merge together. The distributed computation can significantly save computation time as the operations are done on different partitions of the datasets at the same time. Python perform sorting of the entire dataset in memory, without parallel computation, it is very likely to be slower than Spark sorting'

    
    with open(output_filepath_question_b, 'w') as f:
        json.dump(output2, f)
   
    
    
    #write to file
    with open(output_filepath_question_a, 'w') as f:
        f.write("city,stars\n")
        for pair in m1Data:
            f.write("%s,%.1f\n" % (pair[0], pair[1]))
        
    f.close()
    
    
  