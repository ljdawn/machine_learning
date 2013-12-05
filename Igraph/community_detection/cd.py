import igraph as ig
G = ig.Graph.Read_Edgelist('edlist')
p = ig.Graph.pagerank(G)

for vid, pr in enumerate(p):
        print vid, pr

c = ig.Graph.community_label_propagation(G)

for item in c:
	print item