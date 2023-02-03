from pyspark import SparkContext
import sys
import math
import time
from itertools import combinations
import random


start = time.time()
###--------------main----------------

#input_file = '../resource/asnlib/publicdata/yelp_train.csv'
#input_file = 'small_yelp.csv'
output_file = 'result1.txt'


input_file = sys.argv[1]
output_file = sys.argv[2]

sc = SparkContext('local[*]', 'task1')
sc.setLogLevel('WARN')
lines = sc.textFile(input_file)
header = lines.first()
lines = lines.filter(lambda x: x != header)

lines = lines.map(lambda x: x.split(','))

n_users = lines.map(lambda x: x[0]).distinct().collect()
n_bus = lines.map(lambda x: x[1]).distinct().collect()

user_index = lines.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()

n_hash = 20#200
BANDS =  10#100
n_rows=n_hash//BANDS
concept_threshold = (1/BANDS) ** (1/n_rows)
sim_threshold = 0.5
p = 137359
b = 501
m=len(n_users)

def minhash(val_list):
    sig =[]
    for j in range(n_hash):
      
        
        minval = float('inf')
        for i in val_list:
            #val = ((a * i + b) % p) % m
            val = (j * i + b)  % m
            if val < minval:
                minval = val
        sig.append(minval)
    return sig
    

    
##(business_id, user_id_index) -> #('business_id, [0,3,6,8,...]') -> # (business_id, [25,35,1,...]) len = n_hash
print("Start grouping...")   
lines = lines.map(lambda x: (x[1],user_index[x[0]])).groupByKey()
    
print("Start minhashing...")     
sig = lines.mapValues(lambda x: minhash(x)) 
    #.collect()
    
    
signatures_matrix = sig.collect()

print(time.time()-start)


candidates=set()

def permutation(l):
    res = []

    for x in combinations(l,2):
        res.append(x)
    return res
            
def LSH(values):
    bands = []
    for i in range(BANDS):
        start_idx = i*n_rows
        
        band = values[start_idx: start_idx + n_rows]
        
        bands.append(tuple([i,*band]))
    return bands
def f(x): return x      

print("Start LHS...") 
candidates = sig.mapValues(lambda x: LSH(x)).flatMapValues(f).map(lambda x: (x[1], x[0])).groupByKey().filter(lambda x: len(x[1])>1).map(lambda x: permutation(x[1])).flatMap(lambda x :x).distinct()#.collect()
                                                     
print(candidates.collect()[:10])
    
    
    
print(time.time()-start) 
print('n_candidates', candidates.count())
def jaccard_similarity(list1, list2):

    s1 = set(d[list1])
    s2 = set(d[list2])

    sim = float(len(s1.intersection(s2)) / len(s1.union(s2)))
    #print('s1,s2,sim: ', s1,s2,sim)
    return (sorted(tuple([list1, list2])), sim)

d = lines.mapValues(list).collectAsMap()
print('Start similarity...')

#rdd = sc.parallelize(candidates).map(lambda x: jaccard_similarity(x[0],x[1])).sortByKey().collect()
similarities = candidates.map(lambda x: jaccard_similarity(x[0],x[1])).filter(lambda x: x[1]>=sim_threshold).sortByKey().collect()


with open(output_file,'w') as f:
    #f.write('n_candiates: '+ str(candidates.count()))
    #f.write("\n")
    f.write("business_id_1,business_id_2, similarity")
    f.write("\n")
    
    for x in similarities:
        f.write("%s,%s,%.3f\n" % (x[0][0],x[0][1],x[1]))


            

end = time.time()
print("#########/nDuration: ", end-start)
