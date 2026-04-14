from collections import deque
from collections import defaultdict
from collections import Counter

def get_edges(faces):
    edges = []
    for face in faces:
        n = len(face)

        for i in range(n):
            v1 = face[i]
            v2 = face[(i+1) % n]

            edge = tuple(sorted((v1, v2)))
            edges += [edge]
    
    return edges

def get_boundary_edges(edges:set):
    # If edge only present once in the list then, it is involved only in a single face
    counts = Counter(edges)
    return [e for e in edges if counts[e] == 1]


def get_setOfContinuousEdges(boundary_edges):
    # count number of edges until returning to the same edge
    # return list of counts where length of list is length of continous borders
    edges = [tuple(sorted(e)) for e in boundary_edges]
    adj = defaultdict(list) # list of all boundary vertices connected to any other boundary vertex
    for e in  edges:
        u, v = e
        adj[u].append(e)
        adj[v].append(e)
    
    unused = set(edges)
    loops = []

    for e0 in list(unused):
        if e0 not in unused:
            continue
        loop = []

        start_u, start_v = e0
        current_vertex = start_v
        start_vertex = start_u
        prev_edge = None

        unused.remove(e0)
        loop.append(e0)

        while True:
            candidates = [e for e in adj[current_vertex] if e in unused]
            if not candidates:
                break
            e_next = candidates[0]
            unused.remove(e_next)
            loop.append(e_next)
            u, v = e_next
            current_vertex = v if current_vertex == u else u

            if current_vertex == start_vertex:
                break
        
        loops.append(loop)
    loops.sort(key=len, reverse=True)
    
    return loops
