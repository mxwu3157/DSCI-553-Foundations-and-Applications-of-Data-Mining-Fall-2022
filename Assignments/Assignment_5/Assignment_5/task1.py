from pyspark import SparkContext
import sys
import math
import time
from blackbox import BlackBox
import binascii
import random

# int(binascii.hexlify(s.encode('utf8')),16)




t0 = time.time()
#############################
def get_hash_functions(num_hash,a,b):
    
    hash_function_list=[]
    for i in range(num_hash):
        def hash_f(i, user_int):
            return (a[i]*user_int + b[i]) % len_array

        hash_function_list.append(hash_f) 
    return hash_function_list
    
def myhashs(s):
    user_int = int(binascii.hexlify(s.encode('utf8')),16)
    result = []
    for i,f in enumerate(hash_function_list):
        result.append(f(i, user_int))
    return result
############################


file_name = 'testdata.txt'
file_name = '../resource/asnlib/publicdata/users.txt'
stream_size = 100
num_of_asks = 50
output_file = 'result1.txt'

file_name =sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_file = sys.argv[4]



len_array = 69997 

num_hash = math.floor(len_array*math.log(2)/(stream_size*num_of_asks))

print('num_hash = ', num_hash)

a = random.sample(range(100000),num_hash)
b = random.sample(range(100000),num_hash)

hash_function_list = get_hash_functions(num_hash,a,b)
bloom_arr = [0]*len_array

sc = SparkContext('local[*]', 'task2')
sc.setLogLevel('WARN')

def set_bloom_arr(bloom_array,l,num_hash):
    rep_check = []
    for user in l:
        cnt = 0
        rep = False
        for i in user:
            if bloom_array[i] == 1:
                cnt +=1
            else:
                bloom_array[i] =1
        if cnt == num_hash:
            rep = True
        rep_check.append(rep)
        
    return bloom_array, rep_check

def calculate_fpr(m,n,k):
    print('m=%.4f n=%.4f k=%.4f res = %.5f' % (m,n,k, (1-(math.e)**(-k*m/n))**k))
    return (1-(math.e)**(-k*m/n))**k
     
        
fpr = []

for j in range(num_of_asks):
    bx = BlackBox()
    stream_users = bx.ask(file_name, stream_size)
    bloom_hashed_batch = sc.parallelize(stream_users).map(myhashs).collect()
    bloom_arr, rep_check = set_bloom_arr(bloom_arr, bloom_hashed_batch,num_hash)
    fpr.append(calculate_fpr( stream_size*(j+1),len_array, num_hash))

print('fpr', fpr)

with open(output_file, 'w') as f:
    f.write('Time,FPR\n')
    for i, rate in enumerate(fpr):
        f.write(str(i)+","+str(round(rate,6))+"\n")
        
f.close()      
print('Duration: ', time.time()-t0)          
        
    
    #check = all([len_array[i] == 1 for i in hashed_values])
        
    




#input_filename = 
#stream_size
#num_of_asks
#output_filename

#input_filename =sys.argv[1]
#stream_size = int(sys.argv[2])
#num_of_asks = int(sys.argv[3])
#output_filename = sys.argv[4]


