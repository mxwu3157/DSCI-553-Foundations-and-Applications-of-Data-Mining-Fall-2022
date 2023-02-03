


#Method Description:
#This recomender system is mainly a model-based systme using XGBoost.Feature combination and feature augmentation are used to create the training dataset. For user statistics, the number of 'fun' , 'cool', 'useful' and other stats that indicated the content of the review is sum up to be one integer. And the average stars and the number of fans , number of useful click, are also selected to be the input column. For business statistics, business stars and ratings , whether is open , and the latitudes and longitude, number of true attribute, number of mentioned attributes, number of categories are selected. The number of customer checked in in different time frame are sumed up and join with business.To optimized the XGBoost model, gridsearch is used to tune the parameters. And the model that gave the best score are used for prediction


#Error Distribution:                
#>=0 and <1 31                                     
#>=1 and <2 732                         
#>=2 and <3 12409                        
#>=3 and <4 79255                         
#>=4 49617        

#RMSE:
#0.9782842755682551 

#Duration:
#719.0682349205017  s


#---------------Script---------------------

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
import datetime


folder_path = '../resource/asnlib/publicdata/'
test_file_name = '../resource/asnlib/publicdata/yelp_val.csv'
output_file_name = 'result_xgb.csv'

#folder_path = sys.argv[1]
#test_file_name = sys.argv[2]
#output_file_name = sys.argv[3]


t0 = time.time()

sc = SparkContext('local[*]', 'task2.2')

sc.setLogLevel('ERROR')

####-------CF-------------
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
    
    neighborid_weight = [x for x in item_corr.items() if x[1]>0.5]
    
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
    

#####XGB---------------


def flatF(a):
    
    return [a[0], *[x  for ele in a[1] for x in ele ]]

def makeKey(index, l):
    return (l[index], [l[i] for i in range(len(l)) if i!=index ])


    
def getNumCheckin(arr):
    return sum(arr.values())
def reformatJoin(l1, l2):
    if not l2:
        l2 = [0]
    return l1+l2
    
def format_None(s):
    if not s:
        return 0
    return float(s)
def countFriends(s):
    if s == 'None':
        return 0
    return len(s.split(','))
def sumUserStats(l1):
    return sum(l1)


def dateDiff(s):
    d1 = datetime.datetime.strptime(s, '%Y-%m-%d')
    d2 = datetime.datetime.now()

    return abs((d1 - d2).days)
def getNTrueAttributes(d):
    cnt = 0
    if not d:
        return 0
    for k, value in d.items():
        if value=='True':
            cnt+=1
        elif type(value) == dict:
            for _, v in value.items():
                if v == 'True':
                    cnt +=1
        
    return cnt

def getNAttributes(d):
    if not d:
        return 0
    return len(d.keys())

def getNCategories(s):
    if not s: return 0
    if s in ['none','None']: return 0
    
    return len(s.split(s))

    
businessRDD = sc.textFile(folder_path + 'business.json').map(json.loads) \
    .map(lambda x: (x['business_id'] , [float(x['stars']) ,float(x['review_count']) ,
                                        format_None(x['latitude']),format_None(x['longitude']), 
                                        float(x['is_open']),getNAttributes(x['attributes']),
                                        getNTrueAttributes(x['attributes']),
                                        getNCategories(x['categories']),
                                        #hash(x['postal_code'])
                             
                                       ]))
    
checkinRDD = sc.textFile(folder_path + 'checkin.json').map(json.loads) \
    .map(lambda x: (x['business_id'] ,[getNumCheckin(x['time']) ]))


tipBusinessRDD =  sc.textFile(folder_path + 'tip.json').map(json.loads) \
    .map(lambda x: (x['business_id'] , 1)).reduceByKey(lambda a, b: a+b).mapValues(lambda x: [x])

photoRDD = sc.textFile(folder_path + 'photo.json').map(json.loads) \
    .map(lambda x: (x['business_id'] , 1)).reduceByKey(lambda a, b: a+b).mapValues(lambda x: [x])
    
    

businessRDD = businessRDD.leftOuterJoin(checkinRDD).mapValues(lambda x: reformatJoin(x[0] ,x[1])).leftOuterJoin(photoRDD).mapValues(lambda x: reformatJoin(x[0] ,x[1])).leftOuterJoin(tipBusinessRDD).mapValues(lambda x: reformatJoin(x[0] ,x[1]))
####----------
        
userRDD = sc.textFile(folder_path + 'user.json').map(json.loads) \
    .map(lambda x :(x['user_id'] , [float(x['average_stars']), 
                                    float(x['fans']), 
                                    float(x['review_count']), 
                                    float(x['useful']), 
                                    #dateDiff(x['yelping_since']),
                                    countFriends(x['friends']),
                                    #countFriends(x['elite']),
                                    sumUserStats([float(x['funny']),float(x['cool']) ] ),
                                    sumUserStats([float(x['compliment_hot']),float(x['compliment_more']),float(x['compliment_profile']),float(x['compliment_cute']),
                                                  float(x['compliment_list']),float(x['compliment_note']),float(x['compliment_plain']) ,float(x['compliment_cool']),
                                                 float(x['compliment_funny']) ,float(x['compliment_writer']) ,float(x['compliment_photos'])                                   
                                                 ])   
                                   ]
                   ) 
        )

tipUserRDD =  sc.textFile(folder_path + 'tip.json').map(json.loads) \
    .map(lambda x: (x['user_id'] , 1)).reduceByKey(lambda a, b: a+b).mapValues(lambda x: [x])    

userRDD = userRDD.leftOuterJoin(tipUserRDD).mapValues(lambda x: reformatJoin(x[0] ,x[1]))
    
    
#reviewRDD = sc.textFile(folder_path + 'review_train.json').map(json.loads) \
#    .map(lambda x: ((x['user_id'], x['business_id']), [float( x['stars'] )]))\
#    .reduceByKey(lambda a,b : a + b).mapValues(lambda x: [sum(x)/len(x)]) 

#train = sc.textFile(folder_path + 'yelp_train.csv')
#header = train.first()
#train = train.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()


    
    
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





print('fea time', time.time()-t0)

#xgb_model = GridSearchCV(xgboost.XGBRegressor(),
#                   {'max_depth':    [5, 10, 25],
#                    'n_estimators': [500, 1000],
#                   'learning_rate': [0.01, 0.05, 0.1],
#                       'subsample': [0.85,1],
#                #'colsample_bytree':[0.85,0.9,1],
#                       'reg_alpha': [0,0.05,0.1, 0.2],
#                   }, 
#                     n_jobs=10,
#                      scoring='neg_mean_squared_error',
#                      verbose=3)
#xgb_model.fit(xtrainRDD, ytrainRDD)
#print('clf.best_score_', math.sqrt(-1*xgb_model.best_score_))
#print('clf.best_params_', xgb_model.best_params_)


xgb_model = xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.1,verbocity=3, max_depth=5, subsample=0.85, colsample_bytree = 0.95) 
xgb_model.fit(xtrainRDD, ytrainRDD, eval_metric = 'rmse')

print(xgb_model)






valid_in = sc.textFile(test_file_name)
header = valid_in.first()
valid_in = valid_in.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()
key_valid_in, x_valid_in, _ = getFeatureMatrix(valid_in, 'test')


y_val_pred = xgb_model.predict(x_valid_in)




###-----CF predict
###------------ CF Prediction----------
print('cf prediction an hybrid...')
testlines = sc.textFile(test_file_name)
header = testlines.first()
testlines = testlines.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()

cf_pred = testlines.map(lambda x: ((x[0],x[1]),(x[0],x[1]))).mapValues(lambda x: initiatePrediction(x[0],x[1]))#.collect()

#(x[0][0] , x[0][1],x[1] )




###---hybrid-------
weight = 0.0
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
for x in final_pred.take(10) :
    print(x)
print(time.time()-t0)



###-----write-----

f = open(output_file_name,'w')
f.write('user_id, business_id, prediction\n')
for i in range(len(final_pred)):
    f.write(str(final_pred[i][0][0]) + ',' + str(final_pred[i][0][1]) + ',' + str(final_pred[i][1])+ '\n')
  

f.close()

print('Duration: ',time.time() - t0)

#f = open(output_file_name,'w')
#f.write('user_id, business_id, prediction\n')
#for i in range(len(y_val_pred)):
#    f.write(str(key_valid_in[i][0]) + ',' + str(key_valid_in[i][1]) + ',' + str(y_val_pred[i])+ '\n')
  

#f.close()

#print('Duration: ',time.time() - t0)




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

sum_val = 0
cnt = 0
error_distribution = [0,0,0,0,0]
for key in y_true.keys():
    val1 = y_true[key]
    val2 = y_pred.get(key,0)
    if 0<=val2<1:
        error_distribution[0]+=1
    elif 1<=val2<2:
        error_distribution[1]+=1
    elif 2<=val2<3:
        error_distribution[2]+=1
    elif 3<=val2<4:
        error_distribution[3]+=1
    else:
        error_distribution[4]+=1
        
        
    sum_val += (val1-val2)**2
    cnt+=1

val_rmse = math.sqrt(sum_val/cnt)
print('val RMSE: ', val_rmse)

print('Error Distribution: ')
print('>=0 and <1', error_distribution[0])
print('>=1 and <2', error_distribution[1])
print('>=2 and <3', error_distribution[2])
print('>=3 and <4', error_distribution[3])
print('>=4', error_distribution[4])


