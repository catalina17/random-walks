import networkx as nx
import numpy as np
import random

from networkx.classes.function import info, number_of_edges, number_of_nodes
from networkx.readwrite.gpickle import read_gpickle
from numpy.random import choice, uniform


def estimate_edges(G, u):
    # Current timestep
    t = 0
    # Current number of returns to start vertex (u)
    k = 0
    # Time of k-th return to start vertex
    Z_total = 0
    # Degree of start vertex
    deg_u = G.degree(u)

    # Estimates of the number of edges, one for each return time
    m_est = []
    vertex = u
    while k < 100000:
        # Simple random walk
        i = random.randint(0, len(G.neighbors(vertex)) - 1)
        vertex = G.neighbors(vertex)[i]

        # Check if we have a new return to start vertex
        if vertex == u:
            Z_total = t
            k += 1

            # Add new estimate of m (number of edges)
            curr_est = float(Z_total * deg_u) / (2 * k)
            if k % 100 == 0:
                print "Estimate of m at return", k, ":", curr_est
            m_est.append(curr_est)

        # Increase timestep
        t += 1

def estimate_nodes(G, u):
    # Current timestep
    t = 0
    # Current number of returns to start vertex (u)
    k = 0
    # Time of k-th return to start vertex
    Z_total = 0
    # Compute weight of start vertex
    w_u = 1.0
    for v in G.neighbors(u):
        w_u += 1.0 / float(G.degree(v))

    # Estimates of the number of nodes, one for each return time
    n_est = []
    vertex = u
    while k < 100000:
        neighbors = G.neighbors(vertex)
        # Compute weighted edges for the random walk from current vertex
        # w(u,v) <- 1 / deg(u) + 1 / deg(v)
        w_vertex = 1.0 / G.degree(vertex)
        p = [w_vertex + (1.0 / G.degree(n)) for n in neighbors]
        s = sum(p)
        p = [el/s for el in p]

        # Choose next vertex
        new_vertex = np.random.choice(neighbors, p=p)
        vertex = new_vertex

        # Check if we have a new return to start vertex
        if vertex == u:
            Z_total = t
            k += 1

            # Add new estimate of m (number of edges)
            curr_est = float(Z_total * w_u) / (2 * k)
            if k % 100 == 0:
                print "Estimate of n at return ", k, ":", curr_est
            n_est.append(curr_est)

        # Increase timestep
        t += 1

def estimate_triangles(G, u, m):
    # Current timestep
    t = 0
    # Current number of returns to start vertex (u)
    k = 0
    # Time of k-th return to start vertex
    Z_total = 0
    # Degree of start vertex
    deg_u = G.degree(u)
    # Triangles containing start vertex
    t_u = nx.triangles(G, u)

    # Pre-compute the weights for all edges for a faster random walk:
    # weight(e) <- 1 + triangles(e)
    """
    edges = G.edges()
    d1 = dict(((x,y), 1.0) for (x,y) in edges)
    d2 = dict(((y,x), 1.0) for (x,y) in edges)
    edge_weights = dict(d1.items() + d2.items())
    for (x,y) in edge_weights:
        edge_weights[(x,y)] += \
            len([n for n in G.neighbors(y) if n in G.neighbors(x)])
    np.save('WRW_t.npy', edge_weights)
    """
    edge_weights = np.load('WRW_t.npy').item()

    # Estimates of the number of triangles, one for each return time
    t_est = []
    vertex = u
    while k < 100000:
        neighbors = G.neighbors(vertex)
        # Gather relevant weighted edges for the random walk
        p = [edge_weights[(x,y)] \
                 for (x,y) in zip([vertex] * len(neighbors), neighbors)]
        s = sum(p)
        p = [el/s for el in p]

        # Choose next vertex
        new_vertex = np.random.choice(neighbors, p=p)
        vertex = new_vertex

        # Check if we have a new return to start vertex
        if vertex == u:
            Z_total = t
            k += 1

            # Add new estimate of m (number of edges)
            curr_est = max(0,
                           float(Z_total * (deg_u + 2 * t_u)) / (6 * k) - \
                           float(m) / 3)
            if k % 100 == 0:
                print "Estimate of t at return", k, ":", curr_est
            t_est.append(curr_est)

        # Increase timestep
        t += 1


if __name__ == '__main__':

    G = read_gpickle('HTCM.gpickle')

    print "Nodes:", G.number_of_nodes()
    print "Edges:", G.number_of_edges()
    print "Triangles:", sum(nx.triangles(G).values()) / 3

    max_deg = 0
    u = 0
    for node in G.nodes():
        if max_deg < G.degree(node):
            max_deg = G.degree(node)
            u = node
    print "Max degree", max_deg,"at node", u, "(belonging to",\
          nx.triangles(G, u), "triangles)"

    #estimate_edges(G, u)
    #estimate_triangles(G, u, G.number_of_edges())
    estimate_nodes(G, u)
