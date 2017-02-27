import networkx as nx
import numpy as np
import random

from networkx.classes.function import info, number_of_edges, number_of_nodes
from networkx.generators.classic import complete_graph, empty_graph
from networkx.readwrite.gpickle import write_gpickle
from numpy.random import choice, uniform


def extended_prefential_attachment(num_nodes, p, r):
    """Returns a random graph according to the Barabasi-Albert preferential
    attachment model with the extension explained by Cooper et al.

    A graph of ``num_nodes`` nodes is grown by attaching new nodes each with
    ``r`` edges that are preferentially attached to existing nodes with high
    degree.

    Parameters
    ----------
    num_nodes : int
        Number of nodes
    p : float
        Probability of doing preferential attachment; with 1 - p, we add an edge
        to a random neighbour.

    Returns
    -------
    G : Graph
    """

    # Add r initial nodes (m0 in barabasi-speak)
    G = complete_graph(r)
    G.name = "extended_barabasi_albert_graph(%s,%s)"%(num_nodes,r)
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = range(0, r) * (r - 1)
    # Start adding the other n-r nodes. The first node is r.
    source = r

    while source < num_nodes:
        # First edge is (source, vertex_chosen_preferentially)
        i = random.randint(0, len(repeated_nodes) - 1)
        x = repeated_nodes[i]
        G.add_edge(source, x)
        repeated_nodes.extend([source, x])

        # Add the remaining r - 1 edges
        for i in range(0, r - 1):
            curr_p = uniform()
            if curr_p <= p:
                # Attach new vertex to an existing vertex (by preferential
                # attachment)
                i = random.randint(0, len(repeated_nodes) - 1)
                target = repeated_nodes[i]
                G.add_edge(source, target)
                repeated_nodes.extend([source, target])
            else:
                # Attach new vertex to random neighbour of x
                i = random.randint(0, len(G.neighbors(x)) - 1)
                target = G.neighbors(x)[i]
                G.add_edge(source, target)
                repeated_nodes.extend([source, target])

        source += 1
        if source % 1000 == 0:
            print(info(G))
    return G

def Google_graph():
    # Data provided in the graph file
    n = 875713
    m = 5105039

    # Parse the edges
    edges = []
    f = open('web-Google.txt', 'r')
    lines = f.readlines()[4:]
    assert len(lines) == m
    for line in lines:
        vs = [int(num) for num in re.findall(r'\d+', line)]
        edges.append((vs[0], vs[1]))

    # Create the Google graph
    G = empty_graph()
    G.add_edges_from(edges)
    # Check graph is correct
    assert G.number_of_edges() == m
    assert G.number_of_nodes() == n

if __name__ == '__main__':

    num_nodes = 600000
    p = 0.6
    r = 3

    G = extended_prefential_attachment(num_nodes, p, r)
    print "Number of edges in G: "   , str(number_of_edges(G)),\
          " , Number of nodes in G: ", str(number_of_nodes(G))

    write_gpickle(G, "HTCM.gpickle")
