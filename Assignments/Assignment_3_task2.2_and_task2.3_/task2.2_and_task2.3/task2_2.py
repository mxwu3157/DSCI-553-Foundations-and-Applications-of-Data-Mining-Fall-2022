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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  GridSearchCV


folder_path = '../resource/asnlib/publicdata/'
test_file_name = '../resource/asnlib/publicdata/yelp_val_in.csv'
output_file_name = 'result2_2.csv'

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]


t0 = time.time()

sc = SparkContext('local[*]', 'task2.2')

sc.setLogLevel('ERROR')


def flatF(a):
    
    return [a[0], *[x  for ele in a[1] for x in ele ]]

def makeKey(index, l):
    return (l[index], [l[i] for i in range(len(l)) if i!=index ])

businessRDD = sc.textFile(folder_path + 'business.json').map(json.loads) \
    .map(lambda x: (x['business_id'] , [float(x['stars']) ,float(x['review_count'])]))
        
userRDD = sc.textFile(folder_path + 'user.json').map(json.loads) \
    .map(lambda x :(x['user_id'] , [float(x['average_stars']), float(x['useful']) ]) )


reviewRDD = sc.textFile(folder_path + 'review_train.json').map(json.loads) \
    .map(lambda x: ((x['user_id'], x['business_id']), [float( x['stars'] )]))\
    .reduceByKey(lambda a,b : a + b).mapValues(lambda x: [sum(x)/len(x)]) 

train = sc.textFile(folder_path + 'yelp_train.csv')
header = train.first()
train = train.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()


    
    
def getFeatureMatrix(RDD, d):
    if d == 'train':
        resultRDD = RDD.map(lambda x: (x[0], [x[1],float(x[2])])) \
            .leftOuterJoin(userRDD) \
            .map(flatF) \
            .map(lambda x: makeKey(1, x)) \
            .leftOuterJoin(businessRDD)\
            .map(flatF) \
            .map(lambda x: [i if i != None else 0   for i in x  ]) \
            .map(lambda x: ((x[1], x[0],x[2]), [x[i] for i in range(3,len(x)) ])).cache()
            
        xresultRDD = resultRDD.map(lambda x: x[1]).collect()
        yresultRDD = resultRDD.map(lambda x: x[0][2]).collect()
        key_result = resultRDD.map(lambda x: x[0]).collect()
        
    elif d == 'test':
        resultRDD = RDD.map(lambda x: (x[0], [x[1],''])) \
            .leftOuterJoin(userRDD) \
            .map(flatF) \
            .map(lambda x: makeKey(1, x)) \
            .leftOuterJoin(businessRDD)\
            .map(flatF) \
            .map(lambda x: [i if i != None else 0   for i in x  ]) \
            .map(lambda x: ((x[1], x[0],x[2]), [x[i] for i in range(3,len(x)) ])).cache()
            
        xresultRDD = resultRDD.map(lambda x: x[1]).collect()
        yresultRDD = resultRDD.map(lambda x: x[0][2]).collect() 
        key_result = resultRDD.map(lambda x: x[0]).collect()
    return key_result, xresultRDD, yresultRDD 
  
    

train = sc.textFile(folder_path + 'yelp_train.csv')
header = train.first()
train = train.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()

_, xtrainRDD, ytrainRDD = getFeatureMatrix(train, 'train')


print('finish feat')
xgb_model = xgboost.XGBRegressor()  
xgb_model.fit(xtrainRDD, ytrainRDD)

print(xgb_model)



valid_in = sc.textFile(test_file_name)
header = valid_in.first()
valid_in = valid_in.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()
key_valid_in, x_valid_in, _ = getFeatureMatrix(valid_in, 'test')

#for i in range(0,10):
#    print('x_valid_in', str(x_valid_in[i]), 'key_valid_in', str(key_valid_in[i]),'y_valid_in' ,str(y_valid_in[i]) )

y_val_pred = xgb_model.predict(x_valid_in)

f = open(output_file_name,'w')
f.write('user_id, business_id, prediction\n')
for i in range(len(y_val_pred)):
    f.write(str(key_valid_in[i][0]) + ',' + str(key_valid_in[i][1]) + ',' + str(y_val_pred[i])+ '\n')
  

f.close()

print('Duration: ',time.time() - t0)

####------------evaluation-----------


y_pred = {}
with open(output_file_name) as f:
    lines = f.readlines()[1:]
    
    for l in lines:
        x = l.strip().split(',')
        y_pred[tuple([x[0],x[1]])] = float(x[2])
f.close()
y_true = {}        
with open(folder_path + 'yelp_val.csv') as f:
    lines = f.readlines()[1:]
    for l in lines:
        x = l.strip().split(',')
        y_true[tuple([x[0],x[1]])] = float(x[2])
f.close()

print('check length', len(y_true) == len(y_pred))
sum_val = 0
cnt = 0
for key in y_true.keys():
    val1 = y_true[key]
    val2 = y_pred.get(key,0)
    sum_val += (val1-val2)**2
    cnt+=1

val_rmse = math.sqrt(sum_val/cnt)
print('val RMSE: ', val_rmse)






### -------------Parameter optimization-----

#print("Parameter optimization")
#totalX = xtrainRDD + xvalidRDD
#totalY = ytrainRDD + yvalidRDD


#clf = GridSearchCV(xgboost.XGBRegressor(n_jobs=1),
#                   {'max_depth': [ 5, 10,50,100],
#                    'n_estimators': [ 200,500, 1000,2000]}, verbose=1, n_jobs=1, scoring = 'neg_mean_squared_error', cv=5)
#clf.fit(xtrainRDD, xvalidRDD)
#print('clf.best_score_', math.sqrt(-1*clf.best_score_))
#print('clf.best_params_', clf.best_params_)

