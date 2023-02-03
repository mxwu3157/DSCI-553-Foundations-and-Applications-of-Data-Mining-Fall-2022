from pyspark import SparkConf, SparkContext
import sys
import math
import time
from itertools import combinations
import random
from operator import add
import sklearn
import pandas as pd
from xgboost import XGBRegressor
import xgboost
import json
from sklearn.metrics import confusion_matrix, mean_squared_error


folder_path = '../resource/asnlib/publicdata/'
output_file_name = 'result2.txt'

log = open('log2_2.txt','a')
log.write('------------------new run-----------------\n')

t0 = time.time()

sc = SparkContext('local[*]', 'task2.3')

sc.setLogLevel('ERROR')
#user_index = lines.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap() 
#business_index = lines.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()

## (business_id, stars, review_count
businessRDD = sc.textFile(folder_path + 'business.json').map(json.loads) \
    .map(lambda x: (x['business_id'] , [float(x['stars']) ,float(x['review_count'])]))
    #.map(lambda x: {x['business_id'] : [x['stars'] , x['review_count']]}).take(10)
        
userRDD = sc.textFile(folder_path + 'user.json').map(json.loads) \
    .map(lambda x :(x['user_id'] , [float(x['average_stars']), float(x['useful']) ]) )
    #.map(lambda x: {x['user_id']:[x['average_stars'], x['useful']]}).take(10)
    #.map(lambda x :(x['user_id'] , (x['average_stars'], x['useful']) ) )

reviewRDD = sc.textFile(folder_path + 'review_train.json').map(json.loads) \
    .map(lambda x: ((x['user_id'], x['business_id']), [float( x['stars'] )]))\
    .reduceByKey(lambda a,b : a + b).mapValues(lambda x: [sum(x)/len(x)]) 
    #.map(lambda x: (x['user_id'], (x['business_id'], x['stars'] )))

for x in reviewRDD.take(10):
    print(x)

def flatF(a):
    
    return [a[0], *[x  for ele in a[1] for x in ele ]]

def makeKey(index, l):
    return (l[index], [l[i] for i in range(len(l)) if i!=index ])


train = sc.textFile(folder_path + 'yelp_train.csv')
train = train.filter(lambda x: x != 'user_id,business_id,stars').map(lambda x: x.split(',')).cache()


# join: [user_id, business_d] + [user_id,average_stars, useful ]
# map(business_d,user_id, average_stars, useful ) join [business_id, stars, review_count]
#map user_id, business_d, average_stars useful, stars, review_count
#map user_id, business_d, average_stars useful, stars, review_count, stars

#trainRDD = train.map(lambda x: (x[0], (x[1],float(x[2])))) \
#    .join(userRDD) \
#    .map(flatF) \
#    .map(lambda x: makeKey(1, x)) \
#    .join(businessRDD) \
#    .map(flatF) \
#    .map(lambda x: ((x[1], x[0]), [x[i] for i in range(2,len(x)) ])) \
#    .join(reviewRDD) \
#    .map(lambda x: [ *[i for i in x[0]], *[*[j for j in x[1][0]], x[1][1]]    ]).cache()
#    
#xtrainRDD = trainRDD.map(lambda x: [x[i] for i in range(3,len(x)) ]).collect()
#ytrainRDD = trainRDD.map(lambda x: x[2]).collect()


print('duration: ', time.time() - t0)

    
#xgb_model = xgboost.XGBRegressor()  
#xgb_model.fit(xtrainRDD, ytrainRDD)

#print(xgb_model)
print('start fitting xgb model', time.time()-t0)

###-------validation------

valid = sc.textFile(folder_path + 'yelp_val.csv')
valid = valid.filter(lambda x: x != 'user_id,business_id,stars').map(lambda x: x.split(',')).cache()
yvalidRDD = valid.map(lambda x: ((x[0], x[1]), ))


validRDD = valid.map(lambda x: (x[0], [x[1]])) \
    .leftOuterJoin(userRDD).map(lambda x: [x[0],*[i for i in x[1][0]], *[i for i in x[1][1]]]) \
    .map(lambda x: makeKey(1, x)) \
    .leftOuterJoin(businessRDD).map(lambda x: [x[0],*[i for i in x[1][0]], *[i for i in x[1][1]]]) \
    .map(lambda x: ((x[1], x[0]), [x[i] for i in range(2,len(x)) ])) \
    .leftOuterJoin(reviewRDD) \
    .map(lambda x: [*[i for i in x[0]], *[j for j in x[1][0] ]+[x[1][1]]]) \
    .map(lambda x: [i if i != None else 0   for i in x  ]) \
    .cache()
print('validRDD.count()',validRDD.count())
print('valid.count()',valid.count())
    
#xvalidRDD = validRDD.map(lambda x: [x[i] for i in range(3,len(x)) ]).collect()
#yvalidRDD = validRDD.map(lambda x: x[2]).collect()

for x in validRDD.take(10):
    print(x)

#y_val_pred = xgb_model.predict(xvalidRDD)
#xgb_model = xgboost.XGBRegressor()  
#xgb_model.fit(xvalidRDD, yvalidRDD)

#RMSE = math.sqrt(mean_squared_error(yvalidRDD, y_val_pred))
#print('validation RMSE', RMSE)

print('duration: ', time.time() - t0)




