from pyspark import SparkConf, SparkContext
import sys
import math
import time
from itertools import combinations
import random
from operator import add
import sklearn
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
import xgboost
import json
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  GridSearchCV


folder_path = '../resource/asnlib/publicdata/'
test_file_name = '../resource/asnlib/publicdata/yelp_val_in.csv'
output_file_name = 'result2_3.csv'

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]


t0 = time.time()

sc = SparkContext('local[*]', 'task2.2')

sc.setLogLevel('ERROR')



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
   
    return (item1, item2, pearson_corr)



####------prediction function-----------


def getNeighborsWeights(userid_p, itemid_p, n ):
    
    corated_items = user2business[userid_p]
    
    
    #item_corr = []
    item_corr =dict()
    for i in corated_items:    
        corr = calcPearson(itemid_p, i)[2]
        item_corr[i] = corr
    
    neighborid_weight = [x for x in item_corr.items() if x[1]>0]
    
    return neighborid_weight
    
   
    


def makePrediction(userid_p, itemid_p):
    neighborid_weight = getNeighborsWeights(userid_p, itemid_p, n_neighbors)
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
    
    if pred<0:
        pred = 0
    if pred>5:
        pred = 5
    return pred



def initiatePrediction(user_p, item_p):
    if user_p in user_index and item_p  in business_index:
        pred = makePrediction(user_index[user_p],business_index[item_p])
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





    
####-----CF model-------


n_neighbors = 2
dcorr = 0.7

log = open('log_task2_1.txt', 'a')
lines = sc.textFile(folder_path + 'yelp_train.csv')
header = lines.first()
lines = lines.filter(lambda x: x != header)
lines = lines.map(lambda x: x.split(',')).cache()



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
def_rating = 3.5
print('def_rating', def_rating)
print('precomputation duration: ', time.time() - t0)  
    
    
    
    


###--------XGB MODEL---------
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

#train = sc.textFile(folder_path + 'yelp_train.csv')
#header = train.first()
#train = train.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()

   
    
def getFeatureMatrix(RDD, d):
    if d == 'train':
        resultRDD = RDD.map(lambda x: (x[0], [x[1],float(x[2]) ])) \
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
  
    
##-------xgb model------
train = sc.textFile(folder_path + 'yelp_train.csv')
header = train.first()
train = train.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()

_, xtrainRDD, ytrainRDD = getFeatureMatrix(train, 'train')


print('finish feat')
xgb_model = xgboost.XGBRegressor()  
xgb_model.fit(xtrainRDD, ytrainRDD)

print(xgb_model)

##-------xgb predict------
print('xgb prediction....')
valid_in = sc.textFile(test_file_name)
header = valid_in.first()
valid_in = valid_in.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()
key_valid_in, x_valid_in, _ = getFeatureMatrix(valid_in, 'test')



y_val_pred = xgb_model.predict(x_valid_in)
print(time.time()-t0)


###------------ CF Prediction----------
print('cf prediction an hybrid...')
testlines = sc.textFile(test_file_name)
header = testlines.first()
testlines = testlines.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()

cf_pred = testlines.map(lambda x: ((x[0],x[1]),(x[0],x[1]))).mapValues(lambda x: initiatePrediction(x[0],x[1]))#.collect()

#(x[0][0] , x[0][1],x[1] )





###---hybrid-------
weight = 0.1
xgb_pred = []
for i in range(len(y_val_pred)):
    xgb_pred.append([key_valid_in[i][0],key_valid_in[i][1] ,y_val_pred[i] ])

xgb_pred = sc.parallelize(xgb_pred).map(lambda x: ((x[0], x[1]), [x[2]]))
final_pred = cf_pred.map(lambda x: ((x[0][0], x[0][1]), [x[1]])) \
    .join(xgb_pred) \
    .reduceByKey(lambda a, b: a+b) \
    .mapValues(lambda x: weight*x[0][0] + (1-weight)*x[1][0]) \
    .collect()
print('####')
#for x in final_pred.take(10) :
#    print(x)
print(time.time()-t0)


####-----write out------




f = open(output_file_name,'w')
f.write('user_id, business_id, prediction\n')
for i in range(len(final_pred)):
    f.write(str(final_pred[i][0][0]) + ',' + str(final_pred[i][0][1]) + ',' + str(final_pred[i][1])+ '\n')
  

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

