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
test_file_name = '../resource/asnlib/publicdata/yelp_val.csv'
output_file_name = 'result_xgb.csv'

#folder_path = sys.argv[1]
#test_file_name = sys.argv[2]
#output_file_name = sys.argv[3]


t0 = time.time()

sc = SparkContext('local[*]', 'task2.2')

sc.setLogLevel('ERROR')


testRDD = sc.textFile(folder_path + "yelp_train.csv").map(lambda x: x.split(',')).sample(False, 0.3, 81).collect()
#validRDD = sc.textFile(folder_path + "yelp_val.csv")

with open("test_data.csv", 'w') as f:
    f.