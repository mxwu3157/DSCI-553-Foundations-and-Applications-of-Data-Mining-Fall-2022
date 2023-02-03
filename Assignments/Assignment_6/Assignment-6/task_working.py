from pyspark import SparkContext
import sys
import math
import time
from itertools import combinations
import random
from collections import defaultdict
import copy
from sklearn.cluster import KMeans


t0=time.time()
input_file  = '../resource/asnlib/publicdata/hw6_clustering.txt'
#input_file = 'testdata.txt'
n_cluster = 10
output_file = 'result.txt'

t0 = time.time()
sc = SparkContext('local[*]', 'task')
sc.setLogLevel('WARN')
lines = sc.textFile(input_file)

n_dim = len(lines.first().split(','))-2


lines = lines.map(lambda x: x.split(',')).map(lambda x: [float(x[i]) for i in range(n_dim+2)])

data_clusters = lines.map(lambda x: (x[0], x[1]))
points_index = lines.map(lambda x:  (tuple(x[2:]), int(x[0]))).collectAsMap()
index_points = lines.map(lambda x:  (int(x[0]), tuple(x[2:]))).collectAsMap()






clustering_result=defaultdict(list)
###########################
#3####  initialize     #####

n_DiscardPoints = 0
n_CSClusters = 0
n_compressioinPoints = 0
n_RetainedPoints = 0

batch_len = math.ceil(len(points_index)/5)
print('batch_len', batch_len)

initial_data = [point for point, index in points_index.items() if index%5==0]

print('len_ini data', len(initial_data))

outliers_kmeans = KMeans(n_clusters=5*n_cluster).fit(initial_data)

outliers_index = sc.parallelize(outliers_kmeans.labels_).zipWithIndex() \
    .groupByKey().mapValues(list) \
    .filter(lambda x : len(x[1])<=1) \
    .flatMap(lambda x: x[1]).collect()
    
outliers = [initial_data[i] for i in outliers_index]
n_outliers = len(outliers)
print('n_outliers', n_outliers)

#Get DS
initial_data_remained = [initial_data[i] for i in range(len(initial_data)) if i not in outliers_index ]
n_DiscardPoints = len(initial_data_remained)
print('n_DiscardPoints', n_DiscardPoints)

DS_kmeans = KMeans(n_clusters= n_cluster).fit(initial_data_remained)
DS_centroids = sc.parallelize(DS_kmeans.cluster_centers_) \
    .map(lambda x: tuple(x)).zipWithIndex() \
    .map(lambda x: (x[1], x[0])).collectAsMap()
print('DS_centroids',DS_centroids)


def get_DS_stat(pointsIndex_ls):
    N = len(pointsIndex_ls)
    points = [initial_data_remained[x] for x in pointsIndex_ls]
    sum_stat =[]
    sqsum_stat = []

    for i in range(n_dim):
        s = sum([x[i] for x in points])
        sum_stat.append(s)
        sq = sum([x[i]**2 for x in points])
        sqsum_stat.append(sq)
    return (N, sum_stat, sqsum_stat)


DS_clusters = sc.parallelize(DS_kmeans.labels_).zipWithIndex() \
    .groupByKey().mapValues(list).mapValues(get_DS_stat).collect()

for index, label in enumerate(DS_kmeans.labels_):
    point = initial_data_remained[index]
    point_idx = points_index[point]
    clustering_result[label].append(point_idx)

#GEt CS RS
if n_outliers>5*n_cluster:
    CS_kmeans = KMeans(n_clusters=5*n_cluster).fit(outliers)
elif  n_cluster<n_outliers<5*n_cluster:
    CS_kmeans = KMeans(n_clusters=n_cluster).fit(outliers)
else:
    #CS_kmeans = KMeans(n_clusters=n_outliers).fit(outliers)
    CS_kmeans = KMeans(n_clusters=2).fit(outliers)
    
#CS_kmeans = KMeans(n_clusters=5*n_cluster).fit(outliers)


cluster_index_group = sc.parallelize(CS_kmeans.labels_).zipWithIndex() \
    .groupByKey().mapValues(list).cache()

#CS cluster label, return  list
CSClusters = cluster_index_group.filter(lambda x : len(x[1])>1).map(lambda x: x[0]).collect()
#CS cluster centroids
CSClusters_centroids = [CS_kmeans.cluster_centers_[i] for i in CSClusters]
print('CSClusters_centroids',CSClusters_centroids)

# CS cluster representation, return N SUM SUMsq in dict
CS_Clusters =  cluster_index_group.filter(lambda x : len(x[1])>1).mapValues(get_DS_stat).collect()
print('CS_Clusters',CS_Clusters)

# number of CS clusters
n_CSClusters = len(CS_Clusters)
print('n_CSClusters',n_CSClusters)




# CS clusters points    
CS_index = cluster_index_group.filter(lambda x : len(x[1])>1) \
    .flatMap(lambda x: x[1]).collect()
n_compressioinPoints = len(CS_index)
print('n_compressioinPoints',n_compressioinPoints)

CS_points = [outliers[i] for i in CS_index]


#RS    
RS_index = cluster_index_group.filter(lambda x : len(x[1])<=1) \
    .flatMap(lambda x: x[1]).collect()
n_RetainedPoints = len(RS_index)
print('n_RetainedPoints',n_RetainedPoints)

RS_points = [outliers[i] for i in RS_index]


########step 7- step 12################

def get_theta(cluster_rep):
    N = cluster_rep[0]
    sum_list = cluster_rep[1]
    sumsq_list = cluster_rep[2]
    theta = [math.sqrt(sumsq_list[i]/N-(sum_list[i]/N)**2) for i in range(n_dim)]
    return theta

def get_MahalanobisDist_cluster(centroid,x,cluster_rep):
    theta = get_theta(cluster_rep)
    m_dist = [((x[i] - centroid[i])/theta[i])**2 for i in range(n_dim)]
    m_dist = math.sqrt(sum(m_dist))
    return m_dist

def update_cluster_summary(cluster_rep, point):
    N = cluster_rep[0]+1
    sum_value = [ cluster_rep[1][i] + point[i] for i in range(n_dim)]
    sqsum_value = [ cluster_rep[2][i] + (point[i])**2 for i in range(n_dim)]
    
    return (N, sum_value, sqsum_value)
    
    
    

dist_threshold = 2*math.sqrt(n_dim)
for _ in range(1,2): ###batch
    data_load = [point for point, index in points_index.items() if index%5==i]
    
    for new_point in data_load: ##each point
        
        #check DS distance
        m_dist_list = []
        for i, cluster in enumerate(DS_clusters):
            m_dist = get_MahalanobisDist_cluster(DS_centroids[i],new_point,cluster )
            m_dist_list.append(m_dist)
        
        ##check if the point can be put in DS
        if min(m_dist_list)< dist_threshold:
            #give the point the cluster label
            #update cluster summary
            #update n_DiscardPoints
            
            c = m_dist_list.index(min(m_dist_list))
            
            point_idx = points_index[new_point]
            clustering_result[point_idx] = c
            
            DS_clusters[c] = update_cluster_summary(DS_clusters[c],new_point )
            
            n_DiscardPoints +=1
        
        #check CS
        elif CSClusters_centroids>0:
            m_dist_list = []
            for i, cluster in enumerate(CS_Clusters):
                m_dist = get_MahalanobisDist_cluster(CSClusters_centroids[i],new_point,cluster )
                m_dist_list.append(m_dist)

            if min(m_dist_list)< dist_threshold:
                c = m_dist_list.index(min(m_dist_list))
                CS_points.append(new_point)
                #n_compressioinPoints +=1
                CS_Clusters[c] = update_cluster_summary(CS_Clusters[c],new_point )
        else:
            RS_points.append(new_point)
    
    #generate new CS and RS
    if n_RetainedPoints>5*n_cluster:
        CS_kmeans = KMeans(n_clusters=5*n_cluster).fit(RS_points)
    elif  n_cluster<n_RetainedPoints<=5*n_cluster:
        CS_kmeans = KMeans(n_clusters=n_cluster).fit(RS_points)
    else:
        CS_kmeans = KMeans(n_clusters=n_RetainedPoints).fit(RS_points)
        

    cluster_index_group = sc.parallelize(CS_kmeans.labels_).zipWithIndex() \
        .groupByKey().mapValues(list).cache()
    
    #get new CS centroids       
    CSClusters = cluster_index_group.filter(lambda x : len(x[1])>1).map(lambda x: x[0]).collect()
    new_CSClusters_centroids = [CS_kmeans.cluster_centers_[i] for i in CSClusters]
    
    #get new CS summary
    new_CS_Clusters =  cluster_index_group.filter(lambda x : len(x[1])>1).mapValues(get_DS_stat).collectAsMap()
    
    #get new CS index and point
    new_CS_index = cluster_index_group.filter(lambda x : len(x[1])>1) \
        .flatMap(lambda x: x[1]).collect()
    new_CS_points = [RS_points[i] for i in new_CS_index]
    
    
    CS_points.extend(new_CS_points)
               
    #try merge CS clusters
    
    CS_Clusters.extend(new_CS_Clusters)
    merge_check = dict()
    
    for 
    
    
  
    
                    
                    
 
            
            
            
            




            
            
            




print('Duration: ', time.time() - t0)

