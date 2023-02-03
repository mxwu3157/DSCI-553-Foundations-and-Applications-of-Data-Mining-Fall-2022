import csv
import json
from pyspark import SparkContext
import sys
import time
import math
from itertools import combinations

#from  task1 import find_freqitemsets,find_newCandidates,allFreqItemsets,get_occurence

###---------Functions-------

def transform(date, cid,pid):
    #cid = int(cid)
    #comb ='-'.join([date, str(cid)])
    
    comb ='-'.join([date, cid.lstrip('0')])
    return (comb, int(pid))

###--------SON functions-------
def find_freqitemsets(baskets, candidates, support):
    print('find_freqitemsets')
    freq_itemsets=[]
    for itemset in candidates:
        count = 0
        for basket in baskets:
            if itemset.issubset(basket):
                count += 1
        if count >= support:
            itemset = set(itemset)
            freq_itemsets.append(itemset)
                #break
    return freq_itemsets

 
def find_newCandidates(freq_itemsets,baskets, candidates,size):
    print('in find_newCandidates')
    new_cand = []
    non_freq = [x for x in candidates if x not in freq_itemsets ]
    
    items = set().union(*freq_itemsets)
    baskets_flat = set().union(*baskets)
    final_items = items & baskets_flat
    
    for itemset in combinations(final_items,size):
        itemset = set(itemset)
        if itemset not in new_cand:
            new_cand.append(itemset)
        

#    for itemset in combinations(items,size):
#        itemset = set(itemset)
#        if itemset not in new_cand:
#            approve = True
#            for no_set in non_freq:
#                if no_set.issubset(itemset):
#                    approve = False
#                    break
#            if approve:
#                new_cand.append(itemset)

    return new_cand


def allFreqItemsets(candidates, baskets, complete_length, support ):
   
    all_freq_itemsets = []
    
    baskets = list(baskets)
    #print('baskets',baskets)
    if len(baskets) == 0: return []
    perc = len(baskets)/complete_length
    support_scaled = math.ceil(support*perc)
    k = 1
  
    
    while True:
        print('k = ',k)
        freq_itemsets = find_freqitemsets(baskets,candidates,support_scaled )
        all_freq_itemsets.extend(freq_itemsets)
    
        candidates = find_newCandidates(freq_itemsets,baskets, candidates,k+1 )
        if not candidates:
            break
        k = k + 1
 
    return all_freq_itemsets 


def get_occurence(itemsets, baskets):
    baskets = list(baskets)
    #print(baskets)
    result =[]
    for itemset in itemsets:
        count =0
        for basket in baskets:
            if itemset.issubset(basket):
                count+=1
        result.append((tuple(sorted(itemset)),count))
    #print(result)
    return result


def write_listSets(data,f):
    if not data: return 
    data = [list(sorted(x)) for x in data]
    len_dict = {}
    for itemset in data:
        l = len(itemset)
        if l not in len_dict.keys():
            len_dict[l]=[]
        len_dict[l].append(itemset)

    max_l = max(len_dict.keys())  

    for i in range(max_l +1):
        if i not in len_dict:
            continue
        same_len = len_dict[i]
        #same_len = sorted(same_len)
        same_len = ["('" + "', '".join(x) + "')" for x in sorted(same_len)]
        print_str = ','.join(same_len)
        f.write(print_str)
        f.write('\n\n')


###-------Main---------

threshold = int(sys.argv[1])
support = int(sys.argv[2])
input_file = sys.argv[3]
output_file = sys.argv[4]


#input_file = '../resource/asnlib/publicdata/ta_feng_all_months_merged.csv'
#input_file = sys.argv[3]
processed_data_file = 'customer_product.csv'
#output_file = 'result2.txt'


sc = SparkContext('local[*]', 'task2')

###----------Preprocessing---------
t1 = time.time()
lines = sc.textFile(input_file)
header = lines.first()
lines = lines.filter(lambda x: x != header)

lines = lines.map(lambda x: x.split(',')).map(lambda x : transform(x[0][1:-1], x[1][1:-1], x[5][1:-1])) 
    

processed = lines.collect()
with open(processed_data_file,'w') as f:
    out=csv.writer(f)
    out.writerow(['DATE-CUSTOMER_ID','PRODUCT_ID'])
    for row in processed:
        out.writerow(row)
        
f.close()        

    
###--------read in and Apply SON Algorithm---------


start = time.time()

#read file in
lines = sc.textFile(processed_data_file,minPartitions=2)
header = lines.first()
lines = lines.filter(lambda x: x != header).map(lambda x: x.split(','))


#case 1 market-basket model
candidates_init = lines.map(lambda x: x[1]).distinct().map(lambda x : {x}).collect()

lines = lines.groupByKey().map(lambda x: set(x[1])).filter(lambda x: len(x) > threshold).cache()


complete_length = lines.count()
t2 = time.time()
print('########################')
print('complete_length', complete_length)
print('loadin time: ', time.time()-start)
    


###-------Phase 1--------
def par_func(x):
    #return sum(ord(ch) for ch in pair[0]) % n_partition# % n_partition
    return sum(x) % n_partition# % n_partition
    
#dataRDD_par = dataRDD.partitionBy(n_partition, par_func).cache()

candidates = lines \
    .mapPartitions(lambda basket_chunk: allFreqItemsets(candidates_init, basket_chunk, complete_length, support)) \
    .map(lambda x: (tuple(sorted(x)),1)) \
    .reduceByKey(lambda a,b: 1) \
    .map(lambda x: set(x[0])) \
    .collect()

    #.sortBy(lambda x: (len(x), x)) \ 
print('########################')
print('phase 1 time: ', time.time()-t2, 's')
### ------Phase 2----------

frequent_itemsets = lines.mapPartitions(lambda baskets: get_occurence(candidates, baskets)) \
    .reduceByKey(lambda a,b: a+b) \
    .filter(lambda x: x[1]>=support) \
    .map(lambda x: set(x[0])) \
    .collect()




print("#####################")
with open(output_file,'w') as f:
    f.write('Candidates:\n')
    write_listSets(candidates,f)

    f.write('Frequent Itemsets:\n')
    write_listSets(frequent_itemsets,f)

    #write_listSets(frequent_itemsets)
    
f.close()
end = time.time()




print("Duration: ", end-start)







