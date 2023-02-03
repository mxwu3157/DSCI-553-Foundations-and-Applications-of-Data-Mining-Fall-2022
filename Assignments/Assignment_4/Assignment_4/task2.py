from pyspark import SparkContext
import sys
import math
import time
from itertools import combinations
import random
from collections import defaultdict
import copy



input_file = 'test_data.csv'
input_file = '../resource/asnlib/publicdata/ub_sample_data.csv'
bwt_output_file = 'result2.txt'
community_output_file = 'com_result2.txt'
threshold = 7


threshold = int(sys.argv[1])
input_file = sys.argv[2]
bwt_output_file = sys.argv[3]
community_output_file = sys.argv[4]


t0 = time.time()
sc = SparkContext('local[*]', 'task2')
sc.setLogLevel('WARN')
lines = sc.textFile(input_file)
header = lines.first()
lines = lines.filter(lambda x: x != header)

lines = lines.map(lambda x: x.split(','))

user_reviews = lines.groupByKey().mapValues(set)


edges = user_reviews.cartesian(user_reviews) \
    .filter(lambda x: x[0]!=x[1]) \
    .map(lambda x: (x[0][0], x[1][0], len(x[0][1].intersection(x[1][1])))) \
    .filter(lambda x: x[2]>=threshold).map(lambda x : (x[0],x[1])).cache()

#src_nodes = edges.map(lambda x: x[0])
#end_nodes = edges.map(lambda x: x[1])
#vertices = src_nodes.union(end_nodes).distinct().collect()#.map(lambda x: (x,))
#vertices = edges.map(lambda x: x[0]).collect()
graph_dict = edges.groupByKey().mapValues(list).collectAsMap()




class TreeNode:
    def __init__(self, name, neighbors):
        self.neighbors = neighbors
        self.name = name
        
        self.edge_credit = 1
        self.node_credit = 0
        self.visited = False
        self.parent = []
    
    def reinitial(self):
        self.edge_credit = 1
        self.node_credit = 0
        self.visited = False
        self.parent = []
        
      
def dfs_tree(root_name,graph_dict, communities):
    root = tree[root_name]
    
    queue = []
    queue.append(root.name)
    root.node_credit = 1
    levels_relation = []
   
    
    while queue:
        levels_relation.append(queue)
        next_level_nodes = []
        #assign all same level node as visited, in case being added to next level
        for x in queue:
            tree[x].visited = True
        for node_name in queue:
            node = tree[node_name]
            for child in node.neighbors:
                if tree[child].visited== False:
                    next_level_nodes.append(child)
                    tree[child].node_credit +=tree[node_name].node_credit
                    tree[child].parent.append(node_name)
        
        if not next_level_nodes:break
        queue = list(set(next_level_nodes))

    reachable = sum(levels_relation,[])
    for group in communities:
        if reachable[0] in group:
            return levels_relation
    communities.append(reachable) 
    return levels_relation
        
    

def edge_betweenness_assign(levels_relation, btw_dict, tree):
    for i in range( len(levels_relation)-1 ):
        curr_level_nodes = levels_relation[len(levels_relation) - 1 - i]
        for node_name in curr_level_nodes:
            for parent_name in tree[node_name].parent:
                p = tree[parent_name].node_credit/tree[node_name].node_credit
                edge_credit =  p* tree[node_name].edge_credit
                tree[parent_name].edge_credit +=edge_credit
                btw_dict[tuple(sorted([parent_name, node_name]))] +=edge_credit
    
def graph_betweenness(graph_dict, tree):
    betweeness_dict = defaultdict(int)
    communities = []
    
    print('start btw calc', time.time())

    for vertice in graph_dict.keys():
        for node in tree.values():
            node.reinitial()
      
        dfs_levels = dfs_tree(vertice,graph_dict, communities)        
        edge_betweenness_assign(dfs_levels, betweeness_dict, tree)


    for key in betweeness_dict.keys():
        betweeness_dict[key] = betweeness_dict[key]/2
        
    betweeness_dict = sc.parallelize(list(betweeness_dict.items())).sortBy(lambda x: (-x[1], x[0])).collect()
    return betweeness_dict, communities


def dfs_tree_spark(root_name,graph_dict):
    root = tree[root_name]
    
    queue = []
    queue.append(root.name)
    root.node_credit = 1
    levels_relation = []
    
    
    tree={}
    for key, value in graph_dict.items():
        node = TreeNode(key, graph_dict[key])
        tree[key] = node  
    
    while queue:
        levels_relation.append(queue)
        next_level_nodes = []
        #assign all same level node as visited, in case being added to next level
        for x in queue:
            tree[x].visited = True
        for node_name in queue:
            node = tree[node_name]
            for child in node.neighbors:
                if tree[child].visited== False:
                    next_level_nodes.append(child)
                    tree[child].node_credit +=tree[node_name].node_credit
                    tree[child].parent.append(node_name)
        
        if not next_level_nodes:break
        queue = list(set(next_level_nodes))

    reachable = sum(levels_relation,[])
    for group in communities:
        if reachable[0] in group:
            return (levels_relation, reachable)
    
    return (levels_relation, _)


def graph_betweenness_spark(graph_dict, tree):
    betweeness_dict = defaultdict(int)
    
    vertices = sc.parallelize(graph_dict.keys()).map(lambda vertice: dfs_tree_spark(vertice,graph_dict))
    print('######vertices')
    print(vertices.collect())
    
#initial
#tree={}
#for key, value in graph_dict.items():
#    node = TreeNode(key, graph_dict[key])
#    tree[key] = node    
    
#calculate btw
#betweeness_dict, communities = graph_betweenness_spark(graph_dict,tree)   


def btw2file(btw_dict):

    with open(bwt_output_file,'w') as f:
        for x in btw_dict:
            f.write(str(x[0])+ ','+  str(round(x[1], 5)) + '\n')
    f.close()    
   
    
    
    


#######




def find_modularity(graph_dict,removed_graph, communities, m):
    modularity = 0

    for group in communities:
        for node1 in group:
            for node2 in group:
                if node1 in graph_dict[node2] and node2 in graph_dict[node1]:
                    A = 1
                else:
                    A = 0
                ki = len(removed_graph[node1])
                kj = len(removed_graph[node2])

                modularity += (A - ki*kj/(2*m))   
    return modularity

def find_singleG_modularity(graph_dict,removed_graph, group, m):
    modularity = 0
    for node1 in group:
        for node2 in group:
            if node1 in graph_dict[node2] and node2 in graph_dict[node1]:
                A = 1
            else:
                A = 0
            ki = len(removed_graph[node1])
            kj = len(removed_graph[node2])

            modularity += (A - ki*kj/(2*m))   
    return modularity
    

def remove_edge(graph, a, b):
    graph[a].remove(b)
    graph[b].remove(a)
    return graph  


#######################

#initial
tree={}
for key, value in graph_dict.items():
    node = TreeNode(key, graph_dict[key])
    tree[key] = node    
    
#calculate btw
betweeness_dict, communities = graph_betweenness(graph_dict,tree)
btw2file(betweeness_dict)

#find moduality
max_modularity = float('-inf')
m = sum([len(x) for x in graph_dict.values()])/2
modularity = sc.parallelize(communities) \
    .map(lambda group: find_singleG_modularity(graph_dict,graph_dict,group,m)) \
    .reduce(lambda a, b: a+b)
modularity = modularity/(2*m)
max_modularity = max(max_modularity, modularity)

#find edges to remove and remove them
#get updated grph
max_btw = betweeness_dict[0][1]
remove_edges = sc.parallelize(betweeness_dict).filter(lambda x: x[1]== max_btw ).collect()
removed_graph = copy.deepcopy(graph_dict)

for edge,_ in remove_edges:
    remove_edge(removed_graph, edge[0], edge[1])

    
print('#####################')    

curr_com = communities
prev_com = communities
############################
#optimize based on modularity        
while True:
    print('#####################')
    
    tree={}
    for key, value in removed_graph.items():
        node = TreeNode(key, removed_graph[key])
        tree[key] = node  
        
    prev_com = curr_com
    betweeness_dict, communities = graph_betweenness(removed_graph,tree)
    curr_com = communities
    
    modularity = sc.parallelize(curr_com) \
        .map(lambda group: find_singleG_modularity(graph_dict,removed_graph,group,m)) \
        .reduce(lambda a, b: a+b)
    modularity = modularity/(2*m)
    
    print('modularity',modularity)
    if modularity<max_modularity:
        break
    max_modularity = max(max_modularity, modularity)
    
    #if this modularitu keep increasing, keep removing
    max_btw = betweeness_dict[0][1]
    remove_edges = sc.parallelize(betweeness_dict).filter(lambda x: x[1]== max_btw ).collect()
    for edge,_ in remove_edges:
        remove_edge(removed_graph, edge[0], edge[1])
    

    
    
print('#######Final result' )   
print('modularity',modularity)
print('max_modularity',max_modularity)

final_com = sc.parallelize(prev_com).map(lambda x: sorted(list(x))).sortBy(lambda x: (len(x), x)).collect()    

with open(community_output_file,'w') as f:
    for x in final_com:
            f.write(str(x)[1:-1] + '\n')
                                                                                                     
                                                                                                     
f.close()


print('Duration: ', time.time() - t0)
print('thredshod',threshold)