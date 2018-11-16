'''
Credit: Yang Qiao (angelinana0408@gmail.com)
'''

from bs4 import BeautifulSoup as Soup
from collections import OrderedDict


def nx_to_gxl(G, graph_id, filename, graph_edgeids='false', \
              graph_edgemode='undirected'):
    # create a new soup to write to gxl file
    tagText = '''
                <!DOCTYPE gxl SYSTEM "http://www.gupro.de/GXL/gxl-1.0.dtd">
                <gxl>
                <graph id="" edgeids="" edgemode="">
                </graph></gxl>'''
    soup = Soup(tagText, "xml")
    graph = soup.findAll('graph')[0]
    graph.attrs['id'] = graph_id
    graph.attrs['edgeids'] = graph_edgeids
    graph.attrs['edgemode'] = graph_edgemode
    # nodes
    error_sen = ""
    nodeAttrName_type = OrderedDict()
    for nodeID in sorted(G.nodes()):
        for k, v in sorted(G.node[nodeID].items()):  # G.node[nodeID] is attr key-value dict
            if (k in nodeAttrName_type) and (nodeAttrName_type[k] == None):
                nodeAttrName_type[k] == None  # do nothing
            elif (k in nodeAttrName_type) and (type(v) != nodeAttrName_type[k]):
                nodeAttrName_type[
                    k] = None  # None -- this attribute will be deleted forever for every node
                error_sen = "Error! Inconsistent data type -- " + "nodeID: " + str(
                    nodeID) + "; Attribute name: " + k + "\n" + \
                            "\t\t   Should be " + str(
                    nodeAttrName_type[k]) + ", but " + str(
                    type(v)) + " is found" + "\n" + \
                            "\t\t   Graph " + graph_id + ": deleted inconsistent attribute " + k + "!"
                print(error_sen)
            elif k not in nodeAttrName_type:
                if isinstance(v, str) or isinstance(v, bool) or \
                        isinstance(v, int) or isinstance(v, float):
                    nodeAttrName_type[k] = type(v)
                else:
                    nodeAttrName_type[k] = None
                    error_sen = "Error! Wrong data type -- " + "nodeID: " + str(
                        nodeID) + "; Attribute name: " + k + "\n" + \
                                "\t\t   Should be int/float/string/bool, but " + str(
                        type(v)) + " is found" + "\n" + \
                                "\t\t   Graph " + graph_id + ": deleted invalid attribute " + k + "!"
                    print(error_sen)
            else:  # valid and consistent
                nodeAttrName_type[k] = type(v)  # do nothing
    for nodeID in sorted(G.nodes()):
        node_tag = soup.new_tag("node", id=nodeID)
        for k, v in sorted(G.node[nodeID].items()):  # G.node[nodeID] is attr
            # key-value dict
            if nodeAttrName_type[k] == None:
                continue
            attr_type = ''
            if isinstance(v, str):
                attr_type = 'string'
            elif isinstance(v, bool):
                attr_type = 'bool'
            elif isinstance(v, int):
                attr_type = 'int'
            elif isinstance(v, float):
                attr_type = 'float'
            attr_tag = soup.new_tag("attr")
            attr_tag.attrs['name'] = k
            type_tag = soup.new_tag(attr_type)
            type_tag.string = str(v)
            attr_tag.append(type_tag)
            node_tag.append(attr_tag)
        graph.append(node_tag)
    # edges
    edgeAttrName_type = OrderedDict()
    for edge in sorted(G.edges()):
        for k, v in sorted(G[edge[0]][edge[1]].items()):  # G[edge[0]][edge[1]] is attr key-value dict
            if (k in edgeAttrName_type) and (edgeAttrName_type[k] == None):
                edgeAttrName_type[k] == None  # do nothing
            elif (k in edgeAttrName_type) and (type(v) != edgeAttrName_type[k]):
                edgeAttrName_type[k] = None
                error_sen = "Error! Inconsistent data type -- " + "start_id: " + str(
                    edge[0]) + "; end_id: " + str(
                    edge[1]) + "; Attribute name: " + k + "\n" + \
                            "\t\t   Should be " + str(
                    edgeAttrName_type[k]) + ", but " + str(
                    type(v)) + " is found" + "\n" + \
                            "\t\t   Graph " + graph_id + ": deleted inconsistent attribute " + k + "!"
                print(error_sen)
            elif k not in edgeAttrName_type:
                if isinstance(v, str) or isinstance(v, bool) or \
                        isinstance(v, int) or isinstance(v, float):
                    edgeAttrName_type[k] = type(v)
                else:
                    edgeAttrName_type[k] = None
                    error_sen = "Error! Wrong data type -- " + "start_id: " + str(
                        edge[0]) + "; end_id: " + str(
                        edge[1]) + "; Attribute name: " + k + "\n" + \
                                "\t\t   Should be int/float/string/bool, but " + str(
                        type(v)) + " is found" + "\n" + \
                                "\t\t   Graph " + graph_id + ": deleted invalid attribute " + k + "!"
                    print(error_sen)
            else:  # valid and consistent
                edgeAttrName_type[k] = type(v)  # do nothing
    for edge in sorted(G.edges()):
        edge_tag = soup.new_tag("edge", to=edge[1])
        edge_tag.attrs['from'] = edge[0]
        for k, v in sorted(G[edge[0]][edge[1]].items()):  # G[edge[0]][edge[1]] is attr key-value dict
            if edgeAttrName_type[k] == None:
                continue
            attr_type = ''
            if isinstance(v, str):
                attr_type = "string"
            elif isinstance(v, bool):
                attr_type = "bool"
            elif isinstance(v, int):
                attr_type = "int"
            elif isinstance(v, float):
                attr_type = "float"
            attr_tag = soup.new_tag("attr")
            attr_tag.attrs['name'] = k
            type_tag = soup.new_tag(attr_type)
            type_tag.string = str(v)
            attr_tag.append(type_tag)
            edge_tag.append(attr_tag)
        graph.append(edge_tag)
    # save to gxl file
    # print("Saving gxl to {}".format(filename))
    with open(filename, 'w') as f:
        for line in soup.prettify():
            f.write(str(line))
    return nodeAttrName_type, edgeAttrName_type
