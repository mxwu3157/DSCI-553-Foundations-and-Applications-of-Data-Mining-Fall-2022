from pyspark import SparkContext
import sys
import math
import time
from itertools import combinations
import random


####-----function ---------
def calcPearson(item1, item2):
    
    #print(item1, item2)
    coUsers = set(business2user[item1]).intersection(set(business2user[item2]))
    if len(coUsers)<=1: return (item1, item2, dcorr)
    item1_U =[]
    item2_U = []
    item1_avg = businessAvgAll[item1]
    item2_avg = businessAvgAll[item2]
    for user in coUsers:
        item1_U.append(user2businessRating[user][item1] - item1_avg)
        item2_U.append(user2businessRating[user][item2] - item2_avg)
        #print(user,item1,item1_avg,user2businessRating[user][item1])
        #print(user,item2,item2_avg,user2businessRating[user][item2])

    numerator = sum([item1_U[i]* item2_U[i] for i in range(len(coUsers)) ])
    if numerator ==0:
         return (item1, item2, dcorr)
    denominator = math.sqrt(sum([item1_U[i]**2 for i in range(len(coUsers))])) * math.sqrt(sum([item2_U[i]**2 for i in range(len(coUsers))]))
    
    if denominator == 0: 
        return (item1, item2, dcorr)
    
    pearson_corr = numerator/denominator
    
    if item1%10000 == 0: print((item1, item2, pearson_corr))
    return (item1, item2, pearson_corr)



###--------------main----------------
t0 = time.time()
input_file = '../resource/asnlib/publicdata/yelp_train.csv'
pred_file = '../resource/asnlib/publicdata/yelp_val_in.csv'
output_file = 'task2_1_predict.txt'


#input_file = 'small_test2.csv'
#pred_file = 'pred_test.csv'
#output_file = 'result2_1.txt'


#input_file = sys.argv[1]
#pred_file = sys.argv[2]
#output_file = sys.argv[3]

dcorr = 0.1

sc = SparkContext('local[*]', 'task1')
sc.setLogLevel('WARN')
lines = sc.textFile(input_file)
header = lines.first()
lines = lines.filter(lambda x: x != header)

lines = lines.map(lambda x: x.split(',')).cache()
pairsRDD = lines.map(lambda x : ((x[0],x[1]), x[2]))


t1 = time.time()
##--------get precomputation-------

print('Start precomputation....')
print('Getting index dict...')
user_index = lines.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap() 
business_index = lines.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()




print('Getting groupby dict...')
lines = lines.map(lambda x: (user_index[x[0]],business_index[x[1]],float(x[2])))

user2business = lines.map(lambda x : (x[0],x[1])).groupByKey().mapValues(list).collectAsMap()

business2user = lines.map(lambda x :(x[1],x[0])).groupByKey().mapValues(list).collectAsMap()
print('duration: ', time.time() - t1)


print('Getting groupby rating dict...')
user2businessRating = lines.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(dict).collectAsMap()
business2userRating = lines.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(dict).collectAsMap()
#print(lines)
print('duration: ', time.time() - t1)

print('Getting item average  dict...')
businessAvgAll = lines.map(lambda x: (x[1],x[2])).groupByKey().mapValues(lambda x: sum(x)/len(x)).collectAsMap()
userAvgAll = lines.map(lambda x: (x[0],x[2])).groupByKey().mapValues(lambda x: sum(x)/len(x)).collectAsMap()
#print('businessAvgAll', businessAvgAll)\
print('duration: ', time.time() - t1)


t2 = time.time()
print('')
#--------similarity computation-----------

print('Getting pairwise siimilarity...')

sim_pairwise_dict=dict()

#unique_business = lines.map(lambda x: x[0]).distinct().collect()

#sim_pairwise = lines.map(lambda x: x[0]).distinct().map(lambda x: [tuple(sorted([x,y])) for y in unique_business if x != y]).flatMap(lambda x: x).distinct().map(lambda x: calcPearson(x[0],x[1]))#.collect()
#sim_pairwise = lines.map(lambda x: (1,x[0])).distinct().sortBy(lambda x: x[1]).groupByKey().mapValues(lambda x: combinations(x,2)).flatMap(lambda x: x[1]).map(lambda x: calcPearson(x[0],x[1]))

#print('sim_pairwise', sim_pairwise.collect())


#sim_pairwise_dict = sim_pairwise.map(lambda x: ((x[0], x[1]),x[2])).collectAsMap()
#print('sim_pairwise', sim_pairwise_dict)
# output: [(item1, item2, sim), ...]


#lines = lines.collect()
print('duration: ', time.time() - t1)

###--------model-----

#del lines

def makePrediction(user_p, item_p):
    
    
    if not user_p or not item_p: return (user_p, item_p,-100)
    
    sim_dict = dict()
    for i in range(len(business_index)):
        #pair = tuple(sorted([item_p, i]))
        #if pair in sim_pairwise_dict.keys():
        #    sim_list.append((i, sim_pairwise_dict[pair]))
        #else:
        sim = calcPearson(item_p, i)[2]
        sim_dict[i] = sim
        #sim_list.append((i, sim))
            
            #sim_pairwise_dict[pair] = sim
            
          
    sim_list_nei = sorted(sim_dict.items(), key=lambda x: -x[1])[:2]
    neighbor_sele = [x[0] for x in sim_list_nei]

    #neighbor_sele = sim_pairwise.filter(lambda x: item_p == x[0] or item_p == x[1]).sortBy(lambda x: -x[2]).take(2)
    #.map(lambda x: x[0] if x[0]!=item_p else x[1]).take(2)
    #neighbor_sele = [x[0] if x[0]!=item_p else x[1] for x in neighbor_sele ]
    


    deno = sum([abs(sim_dict[x]) for x in neighbor_sele ])
    nume = 0
    for x in neighbor_sele:
        if x in user2businessRating[user_p]:
            val = user2businessRating[user_p][x]
        else: val = businessAvgAll[item_p]
        #val = user2businessRating[user_p].get(x,0)
        nume += val* sim_dict[x]
        
    #nume = sum([user2businessRating[user_p][x] * sim_pairwise_dict[tuple(sorted([item_p,x])) ] for x in neighbor_sele])
    
    prediction = nume/deno
    
    return  (user_p, item_p, prediction)



###------------------predict---------


n_neighbors = 2
dcorr = 0.2



def cast(user, busi):
    userc=None
    busic=None
    if user in user_index:
        userc = user_index[user]
    if busi in business_index:
        busic = business_index[busi]
    return userc, busic
        

def main():
    print('Getting predicton data...')

    testlines = sc.textFile(pred_file)
    header = testlines.first()
    testlines = testlines.filter(lambda x: x != header)

    testlines = testlines.map(lambda x: x.split(','))#.collect()

    
    


    print('Getting predictions dict...')
    #result = testlines.map(lambda x: cast(x[0], x[1])).map(lambda x: makePrediction(x[0], x[1]) ).collect()
    
    

    #def write2()
    with open(output_file,'w') as f:

        #for user_p,item_p,pred in result:
        #result=[]

        for user_p, item_p in testlines.collect():
            u,i = cast(user_p, item_p)
            val = makePrediction(u,i) [2]
            #result.append(val)
        
            print((user_p,item_p,val ))
            #pred = makePrediction(user_index[user_p], business_index[item_p])
            f.write('%s,%s,%.5f\n' %(user_p,item_p,val ))



    f.close()   
    te = time.time()
    print('Duration: ', te-t0)

main()