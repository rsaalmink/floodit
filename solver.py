from collections import defaultdict
import heapq
import time
import copy
import floodIt
import tools as ts
import random

def dfs(f):
    print("--- Backtracking dfs ---")
    result = f.dfs(set([f.field_to_cluster[(0, 0)]]), 1, [f.data[0][0]])
    print("--- Found %i solutions ---"%len(result))
    if result:
        print("--- An optimal solution ---")
        (turns, solution) = sorted(result)[0]
        print(turns, solution)
        f.cache.print_cache_stats()
        f.pp()
        print("RUNNING SOLUTION", turns, solution)
        print("Valid?", f.solve(solution, False))

def bfs(f):
    print("--- Backtracking using breath-first-search ---")
    result = f.bfs(set([f.field_to_cluster[(0, 0)]]), 1, [f.data[0][0]])
    print("--- Found %i solutions ---"%len(result))
    if result:
        print("--- An optimal solution ---")
        (turns, solution) = sorted(result)[0]
        print(turns, solution)
        f.cache.print_cache_stats()
        f.pp()
        print("RUNNING SOLUTION", turns, solution)
        print("Valid?", f.solve(solution, False))

def bfs_genetic(f):
    print("--- Backtracking using breath-first-search genetic ---")
    result = f.bfs_genetic(set([f.field_to_cluster[(0, 0)]]), 1, [f.data[0][0]])
    print("--- Found %i solutions ---"%len(result))
    if result:
        print("--- A potential solution ---")
        (turns, solution) = sorted(result)[0]
        print(turns, solution)
        f.cache.print_cache_stats()
        f.pp()
        print("RUNNING SOLUTION", turns, solution)
        print("Valid?", f.solve(solution, False))

def bfs_genetic_and_backtrack_dfs(f):
    print("--- Finding soultion using breath-first-search genetic, then backtrack dfs to validate. ---")
    result = f.bfs_genetic(set([f.field_to_cluster[(0, 0)]]), 1, [f.data[0][0]])
    print("--- Found %i solutions ---"%len(result))
    if result:
        print("--- A potential solution ---")
        (turns, solution) = sorted(result)[0]
        print(turns, solution)
        f.cache.print_cache_stats()
        f.pp()
        print("RUNNING SOLUTION", turns, solution)
        print("Valid?", f.solve(solution, False))
        print("No guaranteed optimal solution in bfs_genetic, so backtrack dfs using current result as upperbound.")

        g = floodIt.floodItField(f.field_size, f.num_colors, field, True, 1, 0)
        g.upper_bound = turns  #the bfs_genetic solution
        result = g.dfs(set([f.field_to_cluster[(0, 0)]]), 1, [f.data[0][0]])
        print("--- Found %i solutions ---"%len(result))
        if result:
            print("--- An optimal solution ---")
            (turns, solution) = sorted(result)[0]
            print(turns, solution)
            f.cache.print_cache_stats()
            f.pp()
            print("RUNNING SOLUTION", turns, solution)
            print("Valid?", f.solve(solution, False))
        else:
            print("bfs_genetic was optimal with #", turns, solution)
            print("RUNNING SOLUTION", turns, solution)
            print("Valid?", f.solve(solution, False))


def recur(f):
    best_move = f.recur(True)
    print(f.subsets_some(best_move,10))

def user(f):
    # play the game yourself from commandline
    next_move, best_move, won = f.current_color, [], -1
    while won < 0:
        last_move = next_move
        next_move = int(input("Move?"))
        won = f.make_move(next_move)
        data_copy = []
        for i in f.data:
            data_copy.append(copy.copy(i))
        for (x,y) in f.posesses:
            data_copy[x][y] = " "
        print("Last move = %i, movecount = %i"%(f.current_color, f.move_count))
        for i in data_copy:
            for j in i:
                print(j, end=' ')
            print()      
        best_move.append(next_move)
    print("#", len(best_move), best_move)

def iterative(f):
    solution = f.iterative()
    print(solution)
    f.solve(solution, True)

def iterative_1(f):
    solution = f.iterative_1()
    print(solution)
    f.solve(solution, True)

def random_moves(f):
    best_move = f.random_moves(10)
    print(best_move)

def greedy_clusters(f):
    f.solve(f.greedy_clusters())

def backedges(f):
    f.backedges()

def nothing(f):
    pass

def shortest_paths_find_scs(f):
    sequences = []
    for i in range(0, f.field_size):
        for j in range(0, f.field_size):
            sequences.append([f.clustercolor[x] for x in f.shortest_path(f.field_to_cluster[(0, 0)], f.field_to_cluster[(i, j)])])
    print("Number of sequences to all vertices =", len(sequences))
    minimal_sequences = ts.remove_redundant_sequences(sequences)
    print("Removed redundant sequences, reduced problem to #sequences =", len(minimal_sequences))
    for s in minimal_sequences:
        print(s)

    result = ts.scs_backtrack_majorities([], minimal_sequences, depth=3)
    print("non-optimal-but-close shortest common supersequence:", len(result), result)
    print("Win? =", f.solve(result, False))

def spanning_trees(f):
    def print_tree(E, node, depth):
        print("  " * depth + "__" + " "*(2-(len(str(node))))+str(node))
        if node in E:
            for n in E[node]:
                print_tree(E, n, depth+1)
        

    def print_mst(f, E):
        for j,row in enumerate(f.clusterdata):
            r = []
            for i,c in enumerate(row):
                if (i > 0) and ((((row[i-1], c) in E or (c, row[i-1]) in E)) or row[i-1] == c):
                    r.append((4-len(str(c)))*"-" + str(c))
                else:
                    r.append((4-len(str(c)))*" " + str(c))
            print("".join(r))
            r = []
            for i,c in enumerate(row):
                if (j < len(f.clusterdata)-1) and (((c,f.clusterdata[j+1][i]) in E or (f.clusterdata[j+1][i],c) in E) or (f.clusterdata[j+1][i] == c)):
                    r.append("   |")
                else:
                    r.append("    ")
            print("".join(r))

     
    def prim(nodes, edges):
        conn = defaultdict(list)
        for n1,n2,c in edges:
            conn[n1].append((c,n1,n2) )
            conn[n2].append((c,n2,n1) )
     
        mst = []
        used = set([nodes[0]])
        usable_edges = conn[nodes[0]][:]
        heapq.heapify(usable_edges)
     
        while usable_edges:
            cost, n1, n2 = heapq.heappop(usable_edges)
            if n2 not in used:
                used.add(n2)
                mst.append((n1, n2, cost))
     
                for e in conn[n2]:
                    if e[2] not in used:
                        heapq.heappush(usable_edges, e)
        return mst


    unique = defaultdict(int)
    number_of_runs = 1
    for n_times in range(number_of_runs):
        nodes = list(set(f.ccc))
        random.shuffle(nodes)
        edges = []
        for v1 in nodes:
            for color in f.ccc[v1]:
                for v2 in f.ccc[v1][color]:
                    weight = v1+color+v2
                    edges.append((v1, v2, weight))
        mst = prim(nodes, edges)
        edges = []
        for m in mst:
            edges.append((m[0], m[1]))

        print_mst(f, edges)
        s = str(sorted(edges))
        unique[s] += 1


        if not n_times % max(1, number_of_runs / 10):
            print("Found", len(unique), "unique spanning trees from root.")


    print("degree of nodes", sum([f.degree(n) for n in nodes]))

def maximalepadsom(f):
    #WIP
    my_field = []
    for i in range(f.field_size):
        my_field.append([-1]*f.field_size)
    current_cluster = f.field_to_cluster[(f.field_size-1,f.field_size-1)]
    print(f.ccc[current_cluster])



# appspot: 14x14 + 25 moves http://floodit.appspot.com/
# lemoda.net: 14x14 + 25 moves https://www.lemoda.net/javascript/flood-it/index.html

# Interesting papers
# http://www.bris.ac.uk/news/2010/6945.html
# http://floodit.cs.bris.ac.uk/
# http://www.cs.bris.ac.uk/~montanar/papers/floodit.pdf
# http://markgritter.livejournal.com/tag/floodit
# http://people.maths.ox.ac.uk/scott/Papers/floodit.pdf
# http://www.update.uu.se/~shikaree/Westling/

def run_algorithm(mode, field):
    # some more less relevant params
    field_size = 20
    num_colors = 6
    lowerbound_effort = 1
    upperbound_effort = 1
    # upperbound_effort = 2  # better with dfs 

    # Instantiate the floodIt field f, on which we will run our algorithms
    f = floodIt.floodItField(field_size, num_colors, field, True, lowerbound_effort, upperbound_effort)
    f.pp()

    start_time = time.time()
    print("Mode =", mode, "=", modes[mode].__name__)
    # dynamically call function
    modes[mode](f)
    print(time.time() - start_time, "seconds")

    # cProfile.run("main(modes[mode], f)")
 


if __name__ == "__main__":
    # Several algorithmic approaches to doing computations on FloodIt puzzles. Choose one to run.
    modes = {
        00   :   user,            # play the game yourself!
        11   :   recur,           # Fast upperbound algorithm.
        20   :   iterative,       # loop through colors [1,2,3,1,2,3,...] if such a move obtains more clusters
        21   :   iterative_1,     # loop through colors [1,2,3,2,1,2,...] if such a move obtains more clusters
        30   :   random_moves,    # try random moves. Repeat for a certain amount of time, return the best found.
        32   :   greedy_clusters, # greedily choose color that would obtains most clusters
        36   :   backedges,       # home-made, roughly ~1.2*OPT, good for finding an upperbound.
        50   :   shortest_paths_find_scs,  # determine shortest path to all (i,j) positions, take Shortest Common Supersequence, which is a valid solution but not optimal
        60   :   dfs,             # Use depth first search approach with pruning, based on some fast upperbound algorithms.        
        70   :   bfs,             # Breath first search, more efficient for small puzzles.
        75   :   bfs_genetic,     # Bfs, but then after having all solutions of length n, keep the best m in Queue according to metric() before proceeding to depth n+1.
        77   :   bfs_genetic_and_backtrack_dfs,   # best solution generally
        80   :   spanning_trees,  # WIP
        95   :   maximalepadsom,  # myea, wanna do sumthing with dynamic programming. Lock predecessors (31 depends on 22,3 etc)
        99   :   nothing
        }


    mode = 60

    # Some interesting intial fields from field_data.ini 
    # field = "h-4-6"           
    # field = "struct_2"
    # field = "h-6-8"
    # field = "h-12-6-97"
    # field = "h-10-9"
    field = "challenge"       # contest online challenge once for C++ programmers
    # field = "markgritter"     # this guy blogged a lot about it, used one of his field data
    # field = "h-14-6-fun"      # this one is not solved optimally by bfs_genetic! opt=20
    # field = "diamonds2"
    # field = "backtrack_beats_bfs-genetic"  # anti-case genetic algorithm. OPT = DFS = 20. bfs-genetic = 21.
    # field = "backtrack_beats_bfs-genetic_2"  # OPT = DFS = 23. bfs-genetic = 24. Beats in solution, not run-time.
    # field = "random"

    run_algorithm(mode, field)

