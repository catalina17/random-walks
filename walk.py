import math
import networkx as nx
import numpy as np
import random
import scipy.sparse as sparse

from networkx.classes.function import info, number_of_edges, number_of_nodes
from networkx.readwrite.gpickle import read_gpickle
from numpy.random import choice, uniform
from scipy.sparse.linalg import eigs
from sklearn.preprocessing import normalize


def estimate_edges(G, u):
    # Current timestep
    t = 0
    # Current number of returns to start vertex (u)
    k = 0
    # Time of k-th return to start vertex
    Z_total = 0
    # Degree of start vertex
    deg_u = G.degree(u)
    # Build the transition matrix P for the simple random walk
    # w(u,v) <- 1
    """
    P = nx.adjacency_matrix(G)
    P = normalize(P, norm='l1', axis=1)
    P = sparse.csr_matrix(P)
    eigenvals, eigenvecs = eigs(P)
    lambda_2 = sorted(eigenvals)[-2]
    print "Lambda_1:", max(eigenvals), "Lambda_2:", lambda_2
    """
    lambda_2 = 0.91063444938
    Z_uu = 1.0 / (1.0 - lambda_2)
    pi_u = float(deg_u) / G.number_of_edges()
    ct_factor = (2 * Z_uu + pi_u - 1.0) / (pi_u ** 2)

    # Estimates of the number of edges, one for each return time
    m_est = []
    # Variance estimates
    m_var = []
    # Start vertex
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
            m_est.append(curr_est)
            # Add new variance estimate_nodes
            curr_var = math.sqrt(float(deg_u ** 2 / (4 * k)) * ct_factor)
            m_var.append(curr_var)
            if k % 100 == 0:
                print "Estimate of m at", k, ":", curr_est, "w/ var:", curr_var

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
    # Build the transition matrix P for the weighted random walk
    # w(u,v) <- 1 / deg(u) + 1 / deg(v)
    """
    P = sparse.lil_matrix((G.number_of_nodes(), G.number_of_nodes()))
    for (u, v) in G.edges():
        P[u,v] = 1.0 / G.degree(u) + 1.0 / G.degree(v)
        P[v,u] = P[u,v]
    P = normalize(P, norm='l1', axis=1)
    P = sparse.csr_matrix(P)
    eigenvals, eigenvecs = eigs(P)
    lambda_2 = sorted(eigenvals)[-2]
    print "Lambda_1:", max(eigenvals), "Lambda_2:", lambda_2
    """
    lambda_2 = 0.933389228491
    Z_uu = 1.0 / (1.0 - lambda_2)
    pi_u = w_u / float(2 * G.number_of_nodes())
    ct_factor = (2 * Z_uu + pi_u - 1.0) / (pi_u ** 2)
    # Pre-compute degrees
    degs = np.empty((G.number_of_nodes(),), dtype=int)
    for node in G.nodes():
        degs[node] = G.degree(node)

    # Estimates of the number of nodes, one for each return time
    n_est = []
    # Variance estimates
    n_var = []
    # Start vertex
    vertex = u
    while k < 100000:
        neighbors = G.neighbors(vertex)
        # Compute weighted edges for the random walk from current vertex
        w_vertex = 1.0 / degs[vertex]
        p = np.array([w_vertex + (1.0 / degs[n]) for n in neighbors])

        # Choose next vertex
        new_vertex = np.random.choice(neighbors, p=p/sum(p))
        vertex = new_vertex

        # Check if we have a new return to start vertex
        if vertex == u:
            Z_total = t
            k += 1

            # Add new estimate of m (number of edges)
            curr_est = float(Z_total * w_u) / (2 * k)
            n_est.append(curr_est)
            # Add new variance estimate_nodes
            curr_var = math.sqrt(float(w_u ** 2 / (4 * k)) * ct_factor)
            n_var.append(curr_var)
            if k % 100 == 0:
                print "Estimate of n at", k, ":", curr_est, "w/ var:", curr_var

        # Increase timestep
        t += 1

def estimate_triangles(G, u):
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

    """
    # Build the transition matrix P
    N = G.number_of_nodes()
    P = sparse.lil_matrix((N, N))
    for (u, v) in edge_weights.keys():
        P[u,v] = edge_weights[(u,v)]
    P = normalize(P, norm='l1', axis=1)
    P = sparse.csr_matrix(P)
    eigenvals, eigenvecs = eigs(P)
    lambda_2 = sorted(eigenvals)[-2]
    """
    lambda_2 = 0.937153558736
    Z_uu = 1.0 / (1.0 - lambda_2)
    m = G.number_of_edges()
    ct_numerator = float(G.degree(u) + 2 * nx.triangles(G, u))
    pi_u = ct_numerator / (2 * m + 6 * sum(nx.triangles(G).values()))
    ct_factor = (2 * Z_uu + pi_u - 1.0) / (pi_u ** 2)

    # Estimates of the number of triangles, one for each return time
    t_est = []
    # Variance estimates
    t_var = []
    # Start vertex
    vertex = u
    while k < 100000:
        neighbors = G.neighbors(vertex)
        # Gather relevant weighted edges for the random walk
        p = np.array([edge_weights[(x,y)] \
                      for (x,y) in zip([vertex] * len(neighbors), neighbors)])

        # Choose next vertex
        new_vertex = np.random.choice(neighbors, p=p/sum(p))
        vertex = new_vertex

        # Check if we have a new return to start vertex
        if vertex == u:
            Z_total = t
            k += 1

            # Add new estimate of m (number of edges)
            curr_est = max(0,
                           float(Z_total * (deg_u + 2 * t_u)) / (6 * k) - \
                           float(m) / 3)
            t_est.append(curr_est)
            # Add new variance estimate_nodes
            curr_var = math.sqrt(float(ct_numerator ** 2 / (36 * k)) *
                                 ct_factor)
            t_var.append(curr_var)
            if k % 100 == 0:
                print "Estimate of t at", k, ":", curr_est, "w/ var:", curr_var

        # Increase timestep
        t += 1


if __name__ == '__main__':

    G = read_gpickle('HTCM.gpickle')

    print "Nodes:", G.number_of_nodes()
    print "Edges:", G.number_of_edges()
    #print "Triangles:", sum(nx.triangles(G).values()) / 3

    max_deg = 0
    u = 0
    for node in G.nodes():
        if max_deg < G.degree(node):
            max_deg = G.degree(node)
            u = node
    print "Max degree", max_deg,"at node", u, "(belonging to",\
          nx.triangles(G, u), "triangles)"

    #estimate_edges(G, u)
    #estimate_nodes(G, u)
    estimate_triangles(G, u)
