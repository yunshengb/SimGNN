from chorus.model.graphmol import Compound
from chorus.mcsdr import from_array,comparison_array
from chorus.model.atom import Atom
from chorus.model.bond import Bond
from chorus import descriptor
import networkx as nx

# print(nx.__version__)

g1 = nx.read_gexf('temp_1.gexf')
g2 = nx.read_gexf('temp_2.gexf')

c1 = Compound()
c2 = Compound()

for node in g1.nodes():
    c1.add_atom(node, Atom(g1.node[node]['type']))
for edge in g1.edges():
    b = Bond()
    b.order = int(g1[edge[0]][edge[1]]['valence'])
    c1.add_bond(edge[0],edge[1],b)
for node in g2.nodes():
    c2.add_atom(node, Atom(g2.node[node]['type']))
for edge in g2.edges():
    b = Bond()
    b.order = int(g2[edge[0]][edge[1]]['valence'])
    c2.add_bond(edge[0],edge[1],b)

descriptor.assign_valence(c1)
descriptor.assign_valence(c2)

a1 = comparison_array(c1,ignore_hydrogen=False)
a2 = comparison_array(c2,ignore_hydrogen=False)

g = from_array(a1,a2)

f = open('mcs_result.txt','w')
f.write(str(g.edge_count()))
f.close()

 
