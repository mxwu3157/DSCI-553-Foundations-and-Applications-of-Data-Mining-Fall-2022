import pyspark 
from pyspark import SparkContext
import json
import sys

import os
import time

if __name__ == '__main__':
    
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-1.8.0-openjdk-amd64'
    

    
    sc = pyspark.SparkContext(appName = 'task2')
    sc.setLogLevel('WARN')
    
    input_file_path = sys.argv[1]
    output_filepath = sys.argv[2]
    n_partition     = int(sys.argv[3])
    
    output = {}
    
    dataRDD = sc.textFile(input_file_path).map(json.loads).map(lambda review: (review['business_id'],1)).cache()
    
    
    #default
    default = {}
    default['n_partition'] = dataRDD.getNumPartitions()
    default['n_items']     = dataRDD.glom().map(len).collect()
    
    t1 = time.time()
    dataRDD.reduceByKey(lambda a,b: a+b).takeOrdered(10, key=lambda x: (-x[1], x[0]))
    t2 = time.time()
    default['exe_time'] = t2 - t1
    
    output['default'] = default
    
    
    #custome
    customized = {}
    
    def par_func(pair):
        return sum(ord(ch) for ch in pair[0]) % n_partition# % n_partition
    
    dataRDD_par = dataRDD.partitionBy(n_partition, par_func).cache()
    
    customized['n_partition'] = dataRDD_par.getNumPartitions()
    customized['n_items']     = dataRDD_par.glom().map(len).collect()
    
    t1 = time.time()
    dataRDD_par.reduceByKey(lambda a,b: a+b).takeOrdered(10, key=lambda x: (-x[1], x[0]))
    t2 = time.time()
    customized['exe_time'] = t2 - t1
    
    output['customized'] = customized
        
    
    
    print('#####################################')

    
    with open(output_filepath, 'w') as f:
        json.dump(output, f)

