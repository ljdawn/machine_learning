import igraph as ig
import networkx as nx

G = nx.gnm_random_graph(100, 200, seed=1)

GI = ig.Graph()
GI.add_vertices(G.nodes())
GI.add_edges(G.edges())

p = ig.Graph.pagerank(GI)

c = ig.Graph.community_label_propagation(GI)
c1 = ig.Graph.community_multilevel(GI)

for item in c1:
	print item