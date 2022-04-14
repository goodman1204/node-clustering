import networkx as nx
import numpy as np
from networkx.generators import erdos_renyi_graph

attribute_size = 1000

def generate_graphs(num_nodes, density):

    G = erdos_renyi_graph(num_nodes,density)

    # features for each node

    fea_matrix = np.random.choice(attribute_size,(num_nodes,10))


    # labels
    label = np.random.choice(6,num_nodes)


    return G, fea_matrix, label

def save_graph(num_nodes,density,G,fea_matrix,label):

    #save edges
    wp = open('synthetic_{}_{}.edge'.format(num_nodes,density),'w')

    wp.write("#Nodes {}\n".format(num_nodes))
    wp.write("#Edges {}\n".format(len(G.edges)))

    for e in G.edges:
        wp.write("{}\t{}\n".format(e[0],e[1]))

    wp.close()

    #save node

    wp = open('synthetic_{}_{}.node'.format(num_nodes,density),'w')

    wp.write("#Nodes {}\n".format(num_nodes))
    wp.write("#Attributes {}\n".format(attribute_size))

    for n in range(num_nodes):
        for attr in fea_matrix[n]:
            wp.write("{}\t{}\n".format(n,attr))
    wp.close()

    #save label

    wp = open('synthetic_{}_{}.label'.format(num_nodes,density),'w')

    for n in label:
        wp.write("{}\n".format(n))

    wp.close()


if __name__=="__main__":
    num_nodes = 1000
    density = 0.1
    G,fea_matrix,label = generate_graphs(num_nodes,density)
    save_graph(num_nodes,density,G,fea_matrix,label)
    print("Done")












