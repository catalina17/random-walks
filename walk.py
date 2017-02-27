import math
import networkx as nx
import numpy as np
import random
import scipy.sparse as sparse
import threading

from networkx.classes.function import info, number_of_edges, number_of_nodes
from networkx.readwrite.gpickle import read_gpickle
from numpy.random import choice, uniform
from scipy.sparse.linalg import eigs
from sklearn.preprocessing import normalize


def estimate_edges(G, u, run):
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
    # Std. dev estimates using Z_uu
    m_std_Zuu = []
    # Return times
    Tus = []
    # Start vertex
    vertex = u
    while k < 100000:
        # Simple random walk
        i = random.randint(0, len(G.neighbors(vertex)) - 1)
        vertex = G.neighbors(vertex)[i]

        # Check if we have a new return to start vertex
        if vertex == u:
            # Add new return time
            Tus.append(t - Z_total)
            Z_total = t
            k += 1

            # Add new estimate of m (number of edges)
            curr_est = float(Z_total * deg_u) / (2 * k)
            m_est.append(curr_est)
            # Add new std. dev
            curr_std_Zuu = math.sqrt(float(deg_u ** 2 / (4 * k)) * ct_factor)
            m_std_Zuu.append(curr_std_Zuu)
            if k % 100 == 0:
                print "Estimate of m at", k, ":", curr_est, "std:", curr_std_Zuu

        # Increase timestep
        t += 1

    m_est = np.array(m_est)
    m_std_Zuu = np.array(m_std_Zuu)
    Tus = np.array(Tus)

    m_est.tofile('m_est_' + str(run) + '.npy')
    m_std_Zuu.tofile('m_std_Zuu_' + str(run) + '.npy')
    Tus.tofile('Tus' + str(run) + '.npy')

def estimate_nodes(G, u, run):
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
    # Variance estimates using Z_uu
    n_std_Zuu = []
    # Return times
    Tus = []
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
            # Add new return time
            Tus.append(t - Z_total)
            Z_total = t
            k += 1

            # Add new estimate of m (number of edges)
            curr_est = float(Z_total * w_u) / (2 * k)
            n_est.append(curr_est)
            # Add new variance estimate_nodes
            curr_std_Zuu = math.sqrt(float(w_u ** 2 / (4 * k)) * ct_factor)
            n_std_Zuu.append(curr_std_Zuu)
            if k % 100 == 0:
                print "Estimate of n at", k, ":", curr_est, "std:", curr_std_Zuu

        # Increase timestep
        t += 1

    n_est = np.array(n_est)
    n_std_Zuu = np.array(n_std_Zuu)
    Tus = np.array(Tus)

    n_est.tofile('n_est_' + str(run) + '.npy')
    n_std_Zuu.tofile('n_std_Zuu_' + str(run) + '.npy')
    Tus.tofile('Tus' + str(run) + '.npy')

def estimate_triangles(G, u, c, run):
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
    # weight(e) <- 1 + c * triangles(e)
    """
    edges = G.edges()
    d1 = dict(((x,y), 1.0) for (x,y) in edges)
    d2 = dict(((y,x), 1.0) for (x,y) in edges)
    edge_weights = dict(d1.items() + d2.items())
    for (x,y) in edge_weights:
        edge_weights[(x,y)] += \
            c * len([n for n in G.neighbors(y) if n in G.neighbors(x)])
    np.save('WRW_t' + str(c) + '.npy', edge_weights)
    """
    edge_weights = np.load('WRW_t' + str(c) + '.npy').item()

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
    print "lambda_2_0.1:", lambda_2
    """
    if c == 1.0:
        lambda_2 = 0.937153558736
    else:
        lambda_2 = 0.913799592366
    Z_uu = 1.0 / (1.0 - lambda_2)
    m = G.number_of_edges()
    ct_numerator = float(deg_u + 2 * c * t_u)
    t_G = sum(nx.triangles(G).values()) / 3
    pi_u = ct_numerator / (2 * m + 6 * c * t_G)
    ct_factor = (2 * Z_uu + pi_u - 1.0) / (pi_u ** 2)

    # Estimates of the number of triangles, one for each return time
    t_est = []
    # Variance estimates
    t_std_Zuu = []
    # Return times
    Tus = []
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
            # Add new return time
            Tus.append(t - Z_total)
            Z_total = t
            k += 1

            # Add new estimate of m (number of edges)
            curr_est = max(0,
                           float(Z_total * (deg_u + 2 * t_u)) / (6 * k) - \
                           float(m) / 3)
            t_est.append(curr_est)
            # Add new variance estimate_nodes
            curr_std_Zuu = math.sqrt(float(ct_numerator ** 2 / (36 * k)) *
                                     ct_factor)
            t_std_Zuu.append(curr_std_Zuu)
            if k % 100 == 0:
                print "Estimate of t at", k, ":", curr_est, "std:", curr_std_Zuu

        # Increase timestep
        t += 1

    t_est = np.array(t_est)
    t_std_Zuu = np.array(t_std_Zuu)
    Tus = np.array(Tus)

    t_est.tofile('t_est_' + str(run) + str(c) + '.npy')
    t_std_Zuu.tofile('t_std_Zuu_' + str(run) + str(c) + '.npy')
    Tus.tofile('Tus' + str(run) + str(c) + '.npy')

def cycle_edges(G, u, run):
    prev_R_u = 0.0
    R_u = 0.0

    # Quantities needed for estimating variance of R_u
    deg_u = G.degree(u)
    pi_u = deg_u / (2.0 * G.number_of_edges())
    lambda_2 = 0.91063444938 # taken from estimate_edges()
    Z_vv = 1.0 / (1.0 - lambda_2)
    ct_factor = (2 * Z_vv + pi_u - 1.0) / (pi_u ** 2)

    k = 0

    # Estimates of the number of edges, one for each return time
    m_est = []
    # Variance Estimates
    m_std = []
    # Start vertex
    vertex = u
    while k < 100000:
        # Simple random walk
        i = random.randint(0, len(G.neighbors(vertex)) - 1)
        vertex = G.neighbors(vertex)[i]
        R_u += 1.0

        # Check if we have a new return to start vertex
        if vertex == u:
            R_u = (prev_R_u * k + R_u) / (k + 1)
            k += 1

            # Add new estimate of m (number of edges)
            curr_est = 0.5 * deg_u * R_u
            m_est.append(curr_est)
            # Add new variance estimate_nodes
            curr_std = math.sqrt(ct_factor / k)
            m_std.append(curr_std)
            if k % 100 == 0:
                print "Estimate of m at", k, ":", curr_est, "w/ std:", curr_std

            prev_R_u = R_u
            R_u = 0.0

    m_est = np.array(m_est)
    m_std = np.array(m_std)
    m_est.tofile('m_est_' + str(run) + '_cycle.npy')
    m_std.tofile('m_std_' + str(run) + '_cycle.npy')

def cycle_nodes(G, u, run):
    prev_R_u = 0.0
    R_u = 0.0
    deg_u = G.degree(u)
    k = 0
    # Pre-compute degrees
    degs = np.empty((G.number_of_nodes(),), dtype=int)
    for node in G.nodes():
        degs[node] = G.degree(node)

    # Estimates of the number of nodes, one for each return time
    n_est = []
    # Start vertex
    vertex = u
    while k < 100000:
        # Simple random walk
        i = random.randint(0, len(G.neighbors(vertex)) - 1)
        vertex = G.neighbors(vertex)[i]
        R_u += 1.0 / float(degs[vertex])

        # Check if we have a new return to start vertex
        if vertex == u:
            R_u = (prev_R_u * k + R_u) / (k + 1)
            k += 1

            # Add new estimate of m (number of edges)
            curr_est = deg_u * R_u
            n_est.append(curr_est)
            if k % 100 == 0:
                print "Estimate of n at", k, ":", curr_est

            prev_R_u = R_u
            R_u = 0.0

    n_est = np.array(n_est)
    n_est.tofile('n_est_' + str(run) + '_cycle.npy')

def cycle_triangles(G, u, run):
    prev_R_u = 0.0
    R_u = 0.0
    deg_u = G.degree(u)
    k = 0
    # Pre-compute degrees
    degs = np.empty((G.number_of_nodes(),), dtype=float)
    for node in G.nodes():
        degs[node] = G.degree(node)
    # Pre-compute triangles
    ts = np.empty((G.number_of_nodes(),), dtype=float)
    for node in G.nodes():
        ts[node] = nx.triangles(G, node)

    # Estimates of the number of edges, one for each return time
    t_est = []
    # Start vertex
    vertex = u
    while k < 100000:
        # Simple random walk
        i = random.randint(0, len(G.neighbors(vertex)) - 1)
        vertex = G.neighbors(vertex)[i]
        R_u += ts[vertex] / degs[vertex]

        # Check if we have a new return to start vertex
        if vertex == u:
            R_u = (prev_R_u * k + R_u) / (k + 1)
            k += 1

            # Add new estimate of m (number of edges)
            curr_est = 1.0 / 3.0 * deg_u * R_u
            t_est.append(curr_est)
            if k % 100 == 0:
                print "Estimate of t at", k, ":", curr_est

            prev_R_u = R_u
            R_u = 0.0

    t_est = np.array(t_est)
    t_est.tofile('t_est_' + str(run) + '_cycle.npy')

def evaluation(G, u):
    """
    # Estimate m - random walk
    jobs = []
    for i in [1,3,5]:
        thread = threading.Thread(target=estimate_edges, args=(G, u, i,))
        jobs.append(thread)
        thread.start()
    # Estimate n - random walk
    jobs = []
    for i in range(1, 6):
        thread = threading.Thread(target=estimate_nodes, args=(G, u, i,))
        jobs.append(thread)
        thread.start()
    # Estimate t - random walk
    jobs = []
    for i in range(1, 6):
        thread = threading.Thread(target=estimate_triangles, args=(G, u, 1.0, i,))
        jobs.append(thread)
        thread.start()
    """

    # Estimate m - cycle formula
    jobs = []
    for i in [1,2,4,6,7]:
        thread = threading.Thread(target=cycle_edges, args=(G, u, i,))
        jobs.append(thread)
        thread.start()
    """
    # Estimate n - cycle formula
    jobs = []
    for i in range(1, 6):
        thread = threading.Thread(target=cycle_nodes, args=(G, u, i,))
        jobs.append(thread)
        thread.start()
    # Estimate t - cycle formula
    jobs = []
    for i in range(1, 6):
        thread = threading.Thread(target=cycle_triangles, args=(G, u, i,))
        jobs.append(thread)
        thread.start()
    """


if __name__ == '__main__':

    # Load generated graph
    G = read_gpickle('HTCM.gpickle')
    # Load Google graph
    #G = read_gpickle('Google.gpickle')

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
    #estimate_triangles(G, u, 0.1)
    #cycle_nodes(G, u)
    #cycle_edges(G, u)
    #cycle_triangles(G, u)
    evaluation(G, u)
