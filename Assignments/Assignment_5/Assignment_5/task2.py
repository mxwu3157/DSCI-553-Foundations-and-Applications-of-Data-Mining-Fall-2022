from pyspark import SparkContext
import sys
import math
import time
from blackbox import BlackBox
import binascii
import random
import statistics


#############################
def get_hash_functions(num_hash,a,b):
    
    hash_function_list=[]
    for i in range(num_hash):
        def hash_f(i, user_int):
            hash_value = (a[i]*user_int + b[i]) % len_array
            hash_value = bin(hash_value)[2:]
            return hash_value

        hash_function_list.append(hash_f) 
    return hash_function_list
    
def myhashs(s):
    user_int = int(binascii.hexlify(s.encode('utf8')),16)
    result = []
    for i,f in enumerate(hash_function_list):
        result.append(f(i, user_int))
    return result

############################


t0 = time.time()

file_name = 'testdata.txt'
file_name = '../resource/asnlib/publicdata/users.txt'
stream_size = 300
num_of_asks = 40
output_file = 'result2.txt'

file_name =sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_file = sys.argv[4]

num_hash =50
group_size = 2

a = random.sample(range(100000),num_hash)
b = random.sample(range(100000),num_hash)

hash_function_list = get_hash_functions(num_hash,a,b)
len_array = 6999 

sc = SparkContext('local[*]', 'task2')
sc.setLogLevel('WARN')

def get_estimate(r):
    return 2**r

def find_n_trailing_zero(s):
    p = len(s)-1
    while s[p] == "0" and p>=0:
        p -=1
    return len(s)-1-p

def find_max_n_zero(bin_list):
    return max([find_n_trailing_zero(x) for x in bin_list])

def to_matrix(l, n):
    return [l[i:i+n] for i in xrange(0, len(l), n)]


estimate=[]

for j in range(num_of_asks):
    bx = BlackBox()
    stream_users = bx.ask(file_name, stream_size)
    
    hashed_values = sc.parallelize(stream_users).map(myhashs).collect()
    
    est_list = []
    for h_idx in range(num_hash):
        bin_list = [hashed_values[i][h_idx] for i in range(stream_size)]
        res = find_max_n_zero(bin_list)
        est_list.append(2**res)
        
    #get estmate
    est_avg = [sum(est_list[i:i+group_size])/group_size for i in range(0,len(est_list),group_size)]
    estimate.append(statistics.median(est_avg))
        
with open(output_file, 'w') as f:
    f.write('Time,Ground Truth,Estimation\n')
    for i, est in enumerate(estimate):
        f.write(str(i)+",300,"+str(int(est))+"\n")
        
f.close()  
print('check: ',sum(estimate)/(num_of_asks*stream_size))
print('Duration: ', time.time()-t0)              
    
    
    
    
    



