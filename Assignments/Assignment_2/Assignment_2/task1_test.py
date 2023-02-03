from pyspark import SparkContext
import sys
import math
import time
from itertools import combinations






###-------Functions------

def find_freqitemsets(baskets, candidates, support):
    freq_itemsets=[]
    for itemset in candidates:
        count = 0
        for basket in baskets:
            if itemset.issubset(basket):
                count += 1
            if count >= support:
                itemset = set(itemset)
                    #itemset = set(sorted(itemset))
                freq_itemsets.append(itemset)
                continue
    return freq_itemsets

 
def find_newCandidates(freq_itemsets,candidates,size):
    
    non_freq = [x for x in candidates if x not in freq_itemsets ]
    new_cand = []
    a = combinations(freq_itemsets,2)
    
    for x in a:
        val = x[0].union(x[1])
        flag = sum([x.issubset(val) for x in non_freq])
        if (not flag) and (val not in new_candidates):
            new_cand.append(val)
    return new_cand
    
#    for i in range(len(freq_itemsets)):
#        for j in range(i+1, len(freq_itemsets)):
#            x = freq_itemsets[i].union(freq_itemsets[j])
#            if len(x)== size and x not in new_cand:
#                #print(x)
#                approve = True
#                for no_set in non_freq:
#                    if no_set.issubset(x):
#                        approve = False
#                        break
#                if approve:
#                    new_cand.append(x)
#    return new_cand


def allFreqItemsets(candidates, baskets, complete_length, support ):
   
    all_freq_itemsets = []
    
    baskets = list(baskets)
    print('baskets',baskets)
    if len(baskets) == 0: return []
    perc = len(baskets)/complete_length
    support_scaled = math.ceil(support*perc)
    k = 1
  
    
    while True:
        freq_itemsets = find_freqitemsets(baskets,candidates,support_scaled )
        all_freq_itemsets.extend(freq_itemsets)
    
        candidates = find_newCandidates(freq_itemsets,candidates,k+1 )
        if not candidates:
            break
        k = k + 1
        
       
        
    return all_freq_itemsets 


def get_occurence(itemsets, baskets):
    baskets = list(baskets)
    print(baskets)
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
    #data = [x[0] for x in data]
    len_dict = {}
    for itemset in data:
        l = len(itemset)
        if l not in len_dict.keys():
            len_dict[l]=[]
        len_dict[l].append(itemset)

    max_l = max(len_dict.keys())  

    for i in range(max_l +1):
        #print(i)
        if i not in len_dict:
            continue
        same_len = len_dict[i]
        #same_len = sorted(same_len)
        same_len = ["('" + "', '".join(x) + "')" for x in sorted(same_len)]
        print_str = ','.join(same_len)
        f.write(print_str)
        f.write('\n\n')


###-----main----- 

# get input 


input_file = '../resource/asnlib/publicdata/small1.csv'
output_file = 'result1.txt'
case_number = 1
support = 10

case_number = int(sys.argv[1])
support = int(sys.argv[2])
input_file = sys.argv[3]
output_file = sys.argv[4]

start = time.time()

sc = SparkContext('local[*]', 'task1')
lines = sc.textFile(input_file)
header = lines.first()
lines = lines.filter(lambda x: x != header)

lines = lines.map(lambda x: x.split(','))


if case_number ==1:
    #candidates = business
    candidates_init = lines.map(lambda x: x[1]).distinct().map(lambda x : {x}).collect()
    #basket = group by users
    lines = lines.groupByKey().map(lambda x: set(x[1]))


elif case_number == 2:
    #candidates = user
    candidates_init = lines.map(lambda x: x[0]).distinct().map(lambda x : {x}).collect()
    
    #basket = group by business
    lines = lines.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: set(x[1]))
    
    
else: print("Case Number Invalid!!")

complete_length = len(lines.collect())
    


###-------Phase 1--------

candidates = lines \
    .mapPartitions(lambda basket_chunk: allFreqItemsets(candidates_init, basket_chunk, complete_length, support)) \
    .map(lambda x: tuple(sorted(x))) \
    .distinct() \
    .map(lambda x: set(x)) \
    .collect()
#    .map(lambda x: (tuple(sorted(x)),1)) \
#    .reduceByKey(lambda a,b: 1) \
#    .map(lambda x: set(x[0])) \
#    .collect()

    #.sortBy(lambda x: (len(x), x)) \    


### ------Phase 2----------

frequent_itemsets = lines.mapPartitions(lambda basket: get_occurence(candidates, basket)) \
    .reduceByKey(lambda a,b: a+b) \
    .filter(lambda x: x[1]>=support) \
    .map(lambda x: set(x[0])) \
    .collect()

for x in candidates:
    print(x)


print("#####################")
with open(output_file,'w') as f:
    f.write('Candidates:\n')
    write_listSets(candidates,f)

    f.write('Frequent Itemsets:\n')
    write_listSets(frequent_itemsets,f)

    
f.close()
end = time.time()
print("Duration: ", end-start)

#for l in lines:
#    print(l)

#w = open(output_file, 'w')
#w.write(str(lines))
#w.close()
