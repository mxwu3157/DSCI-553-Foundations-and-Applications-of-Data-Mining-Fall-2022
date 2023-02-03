from pyspark import SparkConf, SparkContext
import sys
import math
import time
from itertools import combinations
import random
from operator import add


###-----function------



def calcPearson(item1, item2):
    coUsers = set(business2user[item1]).intersection(set(business2user[item2]))
    #?? how to deal with this edgw case??
    if len(coUsers)<1: 
        return (item1, item2, dcorr)
    item1_U =[]
    item2_U = []
    
    item1_cosum = 0
    item2_cosum = 0
    for user in coUsers:
        item1_U.append(business2userRating[item1][user])
        item2_U.append(business2userRating[item2][user])
        
        item1_cosum += business2userRating[item1][user]
        item2_cosum += business2userRating[item2][user]
    item1_avg = item1_cosum/len(coUsers)
    item2_avg = item2_cosum/len(coUsers)
    
    #item1_avg = businessAvgAll[item1]
    #item2_avg = businessAvgAll[item2]
    nume = 0
    deno1 = 0
    deno2 = 0
    
    #?? how to deal with this edgw case??
    for user in coUsers:
        v1 = user2businessRating[user][item1] - item1_avg
        v2 = user2businessRating[user][item2] - item2_avg
        nume += v1*v2
        deno1 += v1**2
        deno2 += v2**2
       
    if deno1 ==0 or deno2 == 0:
        return (item1, item2, dcorr)
    
    pearson_corr = nume/((math.sqrt(deno1))*(math.sqrt(deno2)))
    #if pearson_corr>0:
    #    pearson_corr = pearson_corr**2
    #else:
    #    pearson_corr = -1* pearson_corr**2
        
    
    #if item1%4000 == 0:
        #f.write('neighborid_weight:  %s' % str(neighborid_weight))
        #print('num= %.3f, deno1 = %.3f, deno2 = %.3f, person = %.3f, item1=%s, item2=%s'%(nume,deno1,deno2, pearson_corr, str(item1_U), str(item2_U)))
    #    print('num= %.3f, deno1 = %.3f, deno2 = %.3f, pearson = %.3f'%(nume,deno1,deno2, pearson_corr))
    
    return (item1, item2, pearson_corr)



####------prediction function-----------


#n_neighbors = 50

def getNeighborsWeights(userid_p, itemid_p, n ):
    
    corated_items = user2business[userid_p]
    
    
    #item_corr = []
    item_corr =dict()
    for i in corated_items:    
        corr = calcPearson(itemid_p, i)[2]
        #item_corr.append(corr)
        item_corr[i] = corr
    
    #neighborid_weight = sorted(item_corr.items(), key=lambda x: -x[1])[:n] 
    #neighborid_weight = [x for x in item_corr.items() if x[1]>0]
    neighborid_weight = [x for x in item_corr.items()if x[1] >0.4]
    #print('n2',neighborid_weight)
    
   
   
    #findinf 2 neighbors
    #highest = max(item_corr)
    #highest_idx = item_corr.index(highest)
    
    #item_corr[highest_idx] = float('-inf')
    
    #second_highest = max(item_corr)
    #if second_highest==float('-inf'):
    #    second_highest = highest
    #second_highest_idx = item_corr.index(second_highest)
    
    #neighborid_weight = [(highest_idx,highest ),(second_highest_idx,second_highest)]
    
    #if userid_p%6000 == 0:
        #f.write('neighborid_weight:  %s' % str(neighborid_weight))
    #    print('neighborid_weight', neighborid_weight)
    
    # [(0,0.342352),(87,0.34203709),....]
    
    return neighborid_weight
    
   
    


def makePrediction(userid_p, itemid_p):
    neighborid_weight = getNeighborsWeights(userid_p, itemid_p, n_neighbors)
    #print(userid_p,itemid_p,neighborid_weight)
    deno = 0
    nume = 0
    
    for x in neighborid_weight:
        deno += abs(x[1])
        if x[0] in user2businessRating[userid_p]:
            val = user2businessRating[userid_p][x[0]]
        else: 
            val = businessAvgAll[itemid_p]
        nume += val* x[1]
        
    if nume == 0 :
        return def_rating
    
    pred = float(nume/deno)
    #pred = round(pred)
    
    if pred<0:
        pred = 0
    if pred>5:
        pred = 5
    
   
    
    return pred



def initiatePrediction(user_p, item_p):
    if user_p in user_index and item_p  in business_index:
        pred = makePrediction(user_index[user_p],business_index[item_p])
        #continue
    # new user, use average item rating
    
    elif item_p  in business_index:
        pred = businessAvgAll[business_index[item_p]] 
    # new item, use average user rating
    elif user_p  in user_index:
        pred = userAvgAll[user_index[user_p]] 
    else:
        pred = def_rating
    
    
    #pred = round(pred)   
    #pred = math.floor(pred)
    return pred
    
####-----readin----
t0 = time.time()
input_file = '../resource/asnlib/publicdata/yelp_train.csv'
pred_file = '../resource/asnlib/publicdata/yelp_val.csv'
output_file = 'result_cf.txt'
#n_neighbors = int(sys.argv[1])

#input_file = sys.argv[1]
#pred_file =  sys.argv[2]
#output_file =  sys.argv[3]





sc = SparkContext('local[*]', 'task2')
sc.setLogLevel('ERROR')
lines = sc.textFile(input_file)
header = lines.first()
lines = lines.filter(lambda x: x != header)
lines = lines.map(lambda x: x.split(',')).cache()


    
####-----Prepare model-------


n_neighbors = 10
dcorr = 0.7

log = open('log_task2_1.txt', 'a')



print('Start precomputation....')
user_index = lines.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap() 
business_index = lines.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()

lines = lines.map(lambda x: (user_index[x[0]],business_index[x[1]],float(x[2])))
user2business = lines.map(lambda x : (x[0],x[1])).groupByKey().mapValues(list).collectAsMap()
business2user = lines.map(lambda x :(x[1],x[0])).groupByKey().mapValues(list).collectAsMap()

print('Getting groupby rating dict...')
user2businessRating = lines.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(dict).collectAsMap()
business2userRating = lines.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(dict).collectAsMap()

businessAvgAll = lines.map(lambda x: (x[1],x[2])).groupByKey().mapValues(lambda x: sum(x)/len(x)).collectAsMap()
userAvgAll = lines.map(lambda x: (x[0],x[2])).groupByKey().mapValues(lambda x: sum(x)/len(x)).collectAsMap()


def_rating = lines.map(lambda x: (1,x[2])).reduceByKey(add).collect()[0][1]/lines.count()
def_rating = 3.0
print('def_rating', def_rating)
print('precomputation duration: ', time.time() - t0)  
    
    
    
    
###---------prediction------
testlines = sc.textFile(pred_file)
header = testlines.first()
testlines = testlines.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()


n_neighbors = [10, 20, 30, 40,50]
n_neighbors = 10
result = testlines.map(lambda x: ((x[0],x[1]),(x[0],x[1]))).mapValues(lambda x: initiatePrediction(x[0],x[1])).collect()


with open(output_file,'w') as f:
    f.write('user_id,business_id,stars\n')
    for i, x in enumerate(result):
        f.write('%s,%s,%.1f\n' %(x[0][0] , x[0][1],x[1] ))
        #if i%200 == 0: 
        #    print('predicting ...', i , time.time() - t0, len(item_neighbors_weight_dict))


f.close()
end = time.time()
print('Duration: ', end - t0)
print('n neighbors:', n_neighbors)





####------------evaluation-----------

import pandas as pd
import numpy as np
import math
y_true = []
with open('../resource/asnlib/publicdata/yelp_val.csv') as f:
    lines = f.readlines()[1:]
    
    for l in lines:
        x = l.strip().split(',')[-1]
        y_true.append(float(x))
        
y_pred = []
with open(output_file,'r') as f:
    lines = f.readlines()[1:]
    
    for l in lines:
        x = l.strip().split(',')[-1]
        y_pred.append(float(x))
        
y_true = y_true[:len(y_pred)]
len(y_pred) == len(y_true) 

MSE = np.square(np.subtract(y_true,y_pred)).mean() 

RMSE = math.sqrt(MSE)
print("Root Mean Square Error:", RMSE)




log.write('n_neig = %d, dcorr =%f, def_rating = %f,  RMSE =  %.5f \n'%(n_neighbors,dcorr, def_rating, RMSE))











