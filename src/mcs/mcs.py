import chorus
import networkx as nx

from chorus.demo import MOL
from chorus import v2000reader as reader
from chorus.draw.svg import SVG
mol = reader.mol_from_text(MOL["demo"])
svg = SVG(mol)
svg.contents()
svg.data_url_scheme()
svg.save("demo.svg")



import networkx as nx
g = nx.Graph()
g.add_node('dinghao', value=1)
g.add_node('jeff', value=1)
print('networkx version:', nx.__version__)
print(g.nodes)
