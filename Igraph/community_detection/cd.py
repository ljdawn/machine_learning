import igraph as ig
import networkx as nx

#G = nx.gnm_random_graph(100, 700, seed=1)
G = nx.gnm_random_graph(100, 700)


GI = ig.Graph()
GI.add_vertices(G.nodes())
GI.add_edges(G.edges())

p = ig.Graph.pagerank(GI)

c5 = ig.Graph.community_label_propagation(GI)
c1 = ig.Graph.community_multilevel(GI)
c2 = ig.Graph.community_spinglass(GI)
c4 = ig.Graph.community_infomap(GI)
c3 = ig.Graph.community_leading_eigenvector(GI)

g = 0
#res_dic = {}
#for cls in c2:
#	for ele in cls:
#		res_dic[ele] = g
#	g += 1

#print res_dic

print c1
print c2
print c3
print c4
print c5
