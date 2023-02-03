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

input_file  = sys.argv[1]
n_cluster = int(sys.argv[2])
output_file = sys.argv[3]

t0 = time.time()
sc = SparkContext('local[*]', 'task')
sc.setLogLevel('WARN')
lines = sc.textFile(input_file)

n_dim = len(lines.first().split(','))-2


lines = lines.map(lambda x: x.split(',')).map(lambda x: [float(x[i]) for i in range(n_dim+2)])

data_clusters = lines.map(lambda x: (x[0], x[1]))
points_index = lines.map(lambda x:  (tuple(x[2:]), int(x[0]))).collectAsMap()
index_points = lines.map(lambda x:  (int(x[0]), tuple(x[2:]))).collectAsMap()
print('total points', len(index_points))

print()



clustering_result=defaultdict(list)
###########################
#3####  initialize     #####



batch_len = math.ceil(len(points_index)/5)
print('batch_len', batch_len)

initial_data = [point for point, index in points_index.items() if index%5==0]

print('len_ini data', len(initial_data))

outliers_kmeans = KMeans(n_clusters=4*n_cluster, random_state=0).fit(initial_data)

outliers_index = sc.parallelize(outliers_kmeans.labels_).zipWithIndex() \
    .groupByKey().mapValues(list) \
    .filter(lambda x : len(x[1])<=1) \
    .flatMap(lambda x: x[1]).collect()
    
outliers = [initial_data[i] for i in outliers_index]

n_outliers = len(outliers)
print('n_outliers', n_outliers)

######################################################
#Get DS
data_DS = [initial_data[i] for i in range(len(initial_data)) if i not in outliers_index ]
n_DiscardPoints = len(data_DS)
print('n_DiscardPoints', n_DiscardPoints)

DS_kmeans = KMeans(n_clusters= n_cluster).fit(data_DS)

#1. centroids
DS_centroids = sc.parallelize(DS_kmeans.cluster_centers_) \
    .map(lambda x: tuple(x)).collect()
print('DS_centroids',DS_centroids)

def get_points(points_index,kmeans_input_data):
    
    return [kmeans_input_data[x] for x in points_index]
    
    
def get_cluster_stat(points_index, kmeans_input_data):
    points = get_points(points_index,kmeans_input_data)
    
    N = len(points)
    sum_stat =[]
    sqsum_stat = []

    for i in range(n_dim):
        s = sum([x[i] for x in points])
        sum_stat.append(s)
        sq = sum([x[i]**2 for x in points])
        sqsum_stat.append(sq)
    return (N, sum_stat, sqsum_stat)

#2. rep
DS_clusters_summary = sc.parallelize(DS_kmeans.labels_).zipWithIndex() \
    .groupByKey().mapValues(list).mapValues(lambda x: get_cluster_stat(x, data_DS)).sortByKey().map(lambda x:x[1]).collect()
print('DS_clusters_summary',DS_clusters_summary)
    
def get_data_label(point):
    
    return points_index[point]

#3. points assignments
for i, c in enumerate(DS_kmeans.labels_):
    point = data_DS[i]
    point_label = get_data_label(point)
    clustering_result[c].append(point_label)

    
    
############################################
#GEt CS RS
if n_outliers>=5*n_cluster:
    CS_kmeans = KMeans(n_clusters=5*n_cluster).fit(outliers)
elif  n_cluster<=n_outliers<5*n_cluster:
    CS_kmeans = KMeans(n_clusters=n_outliers).fit(outliers)
else:
    #CS_kmeans = KMeans(n_clusters=n_outliers).fit(outliers)
    CS_kmeans = KMeans(n_clusters=n_outliers).fit(outliers)
    


cluster_index_group = sc.parallelize(CS_kmeans.labels_).zipWithIndex() \
    .groupByKey().mapValues(list).cache()

#1. centroids
#CS cluster label, return  list
CSClusters = cluster_index_group.filter(lambda x : len(x[1])>1).map(lambda x: x[0]).collect()
print('CSClusters', CSClusters)
#CS cluster centroids
CSClusters_centroids = [tuple(CS_kmeans.cluster_centers_[i]) for i in CSClusters]
print('CSClusters_centroids',CSClusters_centroids)

#2. rep
# CS cluster representation, return N SUM SUMsq in dict
CS_clusters_summary =  cluster_index_group.filter(lambda x : len(x[1])>1).mapValues(lambda x: get_cluster_stat(x, outliers)).map(lambda x:x[1]).collect()

print('CS_clusters_summary',CS_clusters_summary)

# number of CS clusters
n_CSClusters = len(CS_clusters_summary)
print('n_CSClusters',n_CSClusters)



#3. points collections by clusters
# CS clusters points    
CS_index = cluster_index_group.filter(lambda x : len(x[1])>1).collect()
CS_points = []
for c, points_idx in CS_index:
    points = [outliers[idx] for idx in points_idx]
    CS_points.append(points)
print('CS_index',CS_index)
print('CS_points',CS_points)



#RS    
RS_index = cluster_index_group.filter(lambda x : len(x[1])<=1) \
    .flatMap(lambda x: x[1]).collect()
n_RetainedPoints = len(RS_index)
print('n_RetainedPoints',n_RetainedPoints)

RS_points = [outliers[i] for i in RS_index]

print('len(CS_clusters_summary)',len(CS_clusters_summary))
print('len(CSClusters_centroids)',len(CSClusters_centroids))
print('len(CS_points)',len(CS_points))







#####################################################
def get_n_DiscardPoints(clusters_rep):
    val = 0
    for x in clusters_rep:
        val+=x[0]
    return val

def get_n_CScluster(c):
    return len(c)

def get_n_compressioinPoints(clusters_rep):
    val = 0
    for x in clusters_rep:
        val+=x[0]
    return val


def get_n_RetainedPoints(p):
    return len(p)

f = open(output_file,'w')

f.write('The intermediate results:\n')
stat = [get_n_DiscardPoints(DS_clusters_summary), get_n_CScluster(CSClusters_centroids), get_n_compressioinPoints(CS_clusters_summary),get_n_RetainedPoints(RS_points)  ]
stat = [str(x) for x in stat]
f.write('Round 1: '+ ','.join(stat) + '\n')



##########################################################
######### Step7-12 #######################################
print('######### Step7-12 #######################################')

def get_theta(cluster_rep):
    N = cluster_rep[0]
    sum_list = cluster_rep[1]
    sumsq_list = cluster_rep[2]
    theta = [math.sqrt(sumsq_list[i]/N-(sum_list[i]/N)**2) for i in range(n_dim)]
    return theta

def get_MahalanobisDist(centroid,x,cluster_rep):
    theta = get_theta(cluster_rep)
    m_dist = [((x[i] - centroid[i])/theta[i])**2 for i in range(n_dim)]
    m_dist = math.sqrt(sum(m_dist))
    return m_dist

  
    
     
def get_MahalanobisDist_clusters(x, centroids, clusters_rep):
     m_dist_list = []
     for i, cluster in enumerate(clusters_rep):
         m_dist = get_MahalanobisDist(centroids[i],x,cluster )
         m_dist_list.append(m_dist)
     return m_dist_list


def update_cluster_summary(cluster_rep, point):
    N = cluster_rep[0]+1
    sum_value = [ cluster_rep[1][i] + point[i] for i in range(n_dim)]
    sqsum_value = [cluster_rep[2][i] + (point[i])**2 for i in range(n_dim)]
    
    return (N, sum_value, sqsum_value)
  
    
    
def update_centroid(new_rep):
    
    return tuple([sum_val/new_rep[0] for sum_val in new_rep[1]])
    

dist_threshold = 2*math.sqrt(n_dim)
for h in range(1,5): ###batch
    data_load = [point for point, index in points_index.items() if index%5==h]
  
    #check DS distance
    for new_point in data_load:
        M_dists = get_MahalanobisDist_clusters(new_point, DS_centroids, DS_clusters_summary)
    
        ##check if the point can be put in DS
        if min(M_dists)< dist_threshold:
            c_idx = M_dists.index(min(M_dists))

            DS_clusters_summary[c_idx] = update_cluster_summary(DS_clusters_summary[c_idx],new_point )
            DS_centroids[c_idx] = update_centroid(DS_clusters_summary[c_idx])
            
            point_label = points_index[new_point]
            
            clustering_result[c_idx].append(point_label)
        
        else:
            if len(CS_clusters_summary)>0:
                M_dists = get_MahalanobisDist_clusters(new_point, CSClusters_centroids, CS_clusters_summary)


                if min(M_dists)< dist_threshold:
                    c_idx = M_dists.index(min(M_dists))
                    CS_clusters_summary[c_idx] = update_cluster_summary(CS_clusters_summary[c_idx],new_point )
                    CSClusters_centroids[c_idx] = update_centroid(CS_clusters_summary[c_idx])
                    CS_points[c_idx].append(new_point)
                else:
                    RS_points.append(new_point)
                    
            else:
                RS_points.append(new_point)
    
    #generate new CS and RS
    if len(RS_points)>1:
        n_RetainedPoints = len(RS_points)
        if n_RetainedPoints>5*n_cluster:
            CS_kmeans = KMeans(n_clusters=5*n_cluster).fit(RS_points)
        elif  n_cluster<n_RetainedPoints<=5*n_cluster:
            CS_kmeans = KMeans(n_clusters=n_cluster).fit(RS_points)
        else:
            CS_kmeans = KMeans(n_clusters=n_RetainedPoints).fit(RS_points)


        cluster_index_group = sc.parallelize(CS_kmeans.labels_).zipWithIndex().groupByKey().mapValues(list).cache()

        #get new CS centroids       
        CSClusters = cluster_index_group.filter(lambda x : len(x[1])>1).map(lambda x: x[0]).collect()
        new_CSClusters_centroids = [CS_kmeans.cluster_centers_[i] for i in CSClusters]

        #get new CS summary
        new_CS_clusters_summary =  cluster_index_group.filter(lambda x : len(x[1])>1) \
            .mapValues(lambda x: get_cluster_stat(x, RS_points)).map(lambda x:x[1]).collect()

        new_CS_index = cluster_index_group.filter(lambda x : len(x[1])>1).collect()
        #get new _CS points
        new_CS_points = []
        for c, points_idx in new_CS_index:
            points = [RS_points[idx] for idx in points_idx]
            CS_points.append(points)

        CSClusters_centroids.extend(new_CSClusters_centroids)

        CS_clusters_summary.extend(new_CS_clusters_summary)

        CS_points.extend(new_CS_points)
        
        
        #reset RS
        RS_index = cluster_index_group.filter(lambda x : len(x[1])<=1).flatMap(lambda x: x[1]).collect()
        new_RS_points = [RS_points[i] for i in RS_index]
        RS_points = new_RS_points


        
    while len(CS_clusters_summary)>1:
        #print('len(CS_clusters_summary)',len(CS_clusters_summary))
        #print('len(CSClusters_centroids)',len(CSClusters_centroids))
        #print('len(CS_points)',len(CS_points))


        dists = []
        pool = []
        for a in range(len(CS_clusters_summary)):
            for b in range(len(CS_clusters_summary)):
                if a!=b:
                    pool.append(tuple([a,b]))
        for p, pair in enumerate(pool):

            cs1, cs2 = pair[0], pair[1]
            if cs1!=cs2:
                dist = get_MahalanobisDist(CSClusters_centroids[cs2],CSClusters_centroids[cs1],CS_clusters_summary[cs2])
                dists.append(dist)

        if min(dists)>=dist_threshold:
            break
        else:
            min_pair_idx = dists.index(min(dists))
            tobe_merge_pair = pool[min_pair_idx]

            cs1, cs2 = tobe_merge_pair[0], tobe_merge_pair[1]

            updated_centroid = tuple([(CS_clusters_summary[cs1][1][k]+ CS_clusters_summary[cs2][1][k])/(CS_clusters_summary[cs1][0]+CS_clusters_summary[cs2][0]) for k in range(n_dim)])
            CSClusters_centroids[cs2] = updated_centroid
            CSClusters_centroids.pop(cs1)
            
            updated_rep =tuple([CS_clusters_summary[cs1][k] + CS_clusters_summary[cs2][k] for k in range(3)])
            CS_clusters_summary[cs2] = updated_rep
            CS_clusters_summary.pop(cs1)

            CS_points[cs2].extend(CS_points[cs1])
            CS_points.pop(cs1)

        print('len(CS_clusters_summary)',len(CS_clusters_summary))
        print('len(CSClusters_centroids)',len(CSClusters_centroids))
        print('len(CS_points)',len(CS_points))   


    
    
    stat = [get_n_DiscardPoints(DS_clusters_summary), get_n_CScluster(CSClusters_centroids), get_n_compressioinPoints(CS_clusters_summary),get_n_RetainedPoints(RS_points)  ]
    stat = [str(x) for x in stat]
    f.write('Round %d: %s\n' %(h+1, ','.join(stat)))
    

    

for i in range(len(CS_clusters_summary)):
    
    M_dist = get_MahalanobisDist_clusters(CSClusters_centroids[i], DS_centroids, DS_clusters_summary)
    
    if min(M_dists)< dist_threshold:
        c_idx = M_dists.index(min(M_dists))
        points_label = [points_index[x] for x in CS_points[i]]
        clustering_result[c_idx].extend(points_label)
        
    else:
        points_label = [points_index[x] for x in CS_points[i]]
        clustering_result[-1].extend(points_label)

        
points_label = [points_index[x] for x in RS_points]
clustering_result[-1].extend(points_label)        
            
        
    
    


    
f.write('\n')
f.write('The clustering results:\n')

#result = dict() 
#for key, value in clustering_result.items():
#    for p in value:
#        result[p] = key

#for k in sorted(result.keys()):
#    f.write('%d,%d\n' %(k, result[k]))
    

for key, value in clustering_result.items():
    print(key)
    for p in value:
        f.write('%d,%d\n' %(p, key))
    
#f.write(str(clustering_result))
    
    
f.close()    
    
    
    
    

print('Duration: ', time.time() - t0)















