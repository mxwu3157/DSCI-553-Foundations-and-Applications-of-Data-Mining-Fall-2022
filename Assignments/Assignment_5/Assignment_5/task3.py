from pyspark import SparkContext
import sys
import math
import time
from blackbox import BlackBox
import random
 



def main(file_name,stream_size,num_of_asks,output_file):
    t0 = time.time()
    #random.seed(553)
    
    bx = BlackBox()
    reservoir = bx.ask(file_name, stream_size)
    sample_size = 100
    n_batch = 100
    

    f = open(output_file,'w')
    f.write('seqnum,0_id,20_id,40_id,60_id,80_id\n')
    f.write(','.join([str(n_batch), reservoir[0],reservoir[20],reservoir[40],reservoir[60],reservoir[80]])+"\n")

    for i in range(num_of_asks-1):
        stream_users = bx.ask(file_name, stream_size)
        
        for j, user in enumerate(stream_users):
            p =  sample_size/(n_batch + j+1)
            #random.seed(553)
            p_rand = random.random()
            if p_rand<(sample_size/(n_batch + j+1)):
                #random.seed(553)
                rep_ind = random.randint(0, 99)
                reservoir[rep_ind] = user
        n_batch+=stream_size
        f.write(','.join([str(n_batch), reservoir[0],reservoir[20],reservoir[40],reservoir[60],reservoir[80]])+'\n')
    f.close()         

    print('Duration: ', time.time() - t0)  

if __name__ == '__main__':
    

    
    file_name = 'testdata.txt'
    file_name = '../resource/asnlib/publicdata/users.txt'
    stream_size = 100
    num_of_asks = 30
    output_file = 'result3.txt'

    file_name =sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file = sys.argv[4]
    random.seed(553)

    
    main(file_name,stream_size,num_of_asks,output_file)