#floodIt
import cache
import configparser
import copy
import datetime
import heapq
import math
import os
import pprint
import queue
import random
import sys
import time
import tools as ts


class floodItField():
    def __init__(self, size, num_colors, construct_field, verbose=True, lowerbound_effort = 1, upperbound_effort = 1):
        assert num_colors > 1 and size > 0
        self.field_size = size
        self.num_colors = num_colors
        self.upper_bound = -1
        self.verbose = verbose
        self.cache = cache.cache()

        # Build of the initial field state
        if construct_field == "none":
            self.data = []
            for i in range(size):
                self.data.append([(i%num_colors)+1]*size)

        elif construct_field == "random":
            if verbose:
                print("? Field = %s, building up random floodItField instance."%(construct_field))
            valid = False
            while not valid:
                self.data = []
                for i in range(size):
                    a_row = []
                    for j in range(size):
                        a_row.append(random.randint(1,num_colors))
                    self.data.append(a_row)

                cur_color = self.data[0][0]
                # check if construct_field is not all same color
                i = 0
                while not valid and i < self.field_size:
                    j = 0
                    while not valid and j < self.field_size:
                        if self.data[i][j] != cur_color:
                            valid = True
                        j += 1
                    i += 1
        else:
            print("Load named field from field_data.ini file if it exists in there.")
            config = configparser.RawConfigParser()
            config.read(r"c:\Rtgr\PythonScripts\floodIt\field_data.ini")
            if construct_field in config.sections():
                if verbose:
                    print("! Found [%s] in %s, rebuilding FloodIt field instance."%(construct_field, "field_data.ini"))
                d = dict(config.items(construct_field))
                raw_data = d['data'].split("\n")
                self.data = []
                for line in raw_data:
                    self.data.append([int(x) for x in line.replace("[", "").replace("]", "").split(",") if x])
                self.field_size = len(self.data)
                try:
                    self.upper_bound = int(d['opt'])
                except:
                    pass
                self.num_colors = len(set([y for x in self.data for y in x]))
            else:
                raise Exception("No field named [%s] found in %s."%(construct_field, "field_data.ini"))


        self.reset(verbose)
        self.lower_bound = self.lowerbound(lowerbound_effort)
        self.upper_bound = self.upperbound(upperbound_effort)
        
        if verbose:
            print("Instantiated field '%s' with field %i size and %i colors, contains %i clusters, [%i <= OPT <= %i]."%(construct_field, self.field_size, self.num_colors, len(self.clusters), self.lower_bound, self.upper_bound))


    def print_clusterdata(self, possesets = []):
        width = len(str(len(self.clusterdata)))+3
        for j,row in enumerate(self.clusterdata):
            r = []
            for i,color in enumerate(row):
                if (color not in possesets):
                    if ((i > 0) and row[i-1] == color):
                        r.append((width-len(str(color)))*"-" + str(color))
                    else:
                        r.append((width-len(str(color)))*" " + str(color))
                else:
                    r.append(width*" ")
            print("".join(r))
            r = []
            if (j < len(self.clusterdata)-1):
                for i,color in enumerate(row):
                    if (self.clusterdata[j+1][i] == color and self.clusterdata[j+1][i] not in possesets):
                        r.append((width-1)*" "+ "|")
                    else:
                        r.append(width*" ")
                print("".join(r))


    def pp(self):
        # 'pretty' prints the fielddata
        if len(self.data) < 6:
            for x in self.data:
                print(x)
        else:
            pprint.pprint(self.data, indent=2)

    
    def reset(self, verbose=False):
        # (re)computes clusters, graphs and all related usefull properties. Some functions require using it
        # recur
        # random_moves
        # solve
        # iterative
        # greedy_clusters        
        self.clusters = self.build_clusters()

        # field_to_cluster maps coordinate to a cluster for lookup
        field_to_cluster = {}
        for i, (color, cluster) in enumerate(self.clusters):
            for field in cluster:
                field_to_cluster[field] = i
        self.field_to_cluster = field_to_cluster

        # Simple graph datastrucutre ccc: "From cluster", c1, "with color", col, "we obtain cluster", c2
        data = tuple(list(map(tuple, self.data)))
        self.ccc = {}
        for i, (color, cluster) in enumerate(self.clusters):
            self.ccc[i] = {}
            # try all but current this.cluster's colors, for find neighbouring self.clusters.
            for c in [x+1 for x in range(self.num_colors) if x+1 != color]:
                borders = set()
                for field in self.clusters[i][1]:
                    for neigh in [x for x in ts.obtain_neighbours_single_color_self_free(data, self.field_size, field[0], field[1], c)]:
                        borders.add(field_to_cluster[neigh])
                if borders:
                    self.ccc[i][c] = borders

        # colorsleft[i] invariantly keeps track of the number of clusters left having color i
        self.colorsleft = [0 for x in range(self.num_colors)]
        for (color, fields) in self.clusters:
            self.colorsleft[color-1] += 1
        self.colorsleft[self.data[0][0]-1] -= 1
        if verbose:
            print("Color count", self.colorsleft)

        self.current_color = self.data[0][0]
        self.posesses = [(0,0)]
        self.move_count = 0
        self.posesses = list(self.clusters[self.field_to_cluster[self.posesses[0]]][1])
        self.clustercolor = {}
        self.clusterdata = []
        for x in range(self.field_size):
            self.clusterdata.append([0 for x in range(self.field_size)])
        for i,(color, fields) in enumerate(self.clusters):
            for f in fields:
                self.clusterdata[f[0]][f[1]] = i
            self.clustercolor[i] = color

        if verbose:
            self.print_clusterdata()
        self.num_clusters = len(self.clusters)
        self.posesses_clusters = [self.field_to_cluster[(0,0)]]
        self.cache.clear_cache()


    def build_clusters(self):
        data = tuple(list(map(tuple, self.data)))
        # group together all positions that are part of the same cluster (== share same color and are horizontal/vertical neighbours)
        all_positions = sorted([(x,y) for x in range(self.field_size) for y in range(self.field_size)], reverse=True)
        clusters = []
        while all_positions:
            # still new clusters to be found
            next_position = all_positions.pop()
            color = data[next_position[0]][next_position[1]]
            # make a new cluster c containing next_position
            c = set()
            c.add(next_position)
            # expand_c contains positions that can be added for this color
            expand_c = set(ts.obtain_neighbours_single_color_self_free(data, self.field_size, next_position[0], next_position[1], color))
            while expand_c:
                n = expand_c.pop()
                c.add(n)
                for ne in ts.obtain_neighbours_single_color_self_free(data, self.field_size, n[0], n[1], color):
                    if ne in all_positions:
                        expand_c.add(ne)
                all_positions.remove(n)
            clusters.append((color, c))
        return clusters


    def lowerbound(self, lowerbound_effort):
        # a collection of algorithms that produce a lower bound on optimal solution OPT

        # Taken from paper  http://www.cs.bris.ac.uk/~montanar/papers/floodit.pdf
        # This is in fact a lower bound: there exists a field that needs at least
        # this amount of moves for given n and c. NB might not hold for CURRENT field.
        def paper2(n,c):
            if c > n**2 or c < 1:
                return 0
            else:
                return ((c-1)**0.5 * n / 2) - (c / 2)

        def longest_shortest_path():
            # For ALL positions, determine shortest path to 0,0
            sequences = []
            for i in range(0, self.field_size):
                for j in range(0, self.field_size):
                    sequences.append(len(self.shortest_path(self.field_to_cluster[(0, 0)], self.field_to_cluster[(i, j)])))
            # longest shortest_path is a lowerbound
            return max(sequences)


        n = self.field_size
        c = len([x for x in self.colorsleft if x])
        lowerbounds = []
        lowerbounds.append(c)
        if lowerbound_effort > 0:
            # lowerbounds.append(paper2(n,c)) # not valid for all fields, but there exists one!
            lowerbounds.append(longest_shortest_path())
            self.reset()
        if self.verbose:
            print("Lowerbound returned", max(lowerbounds), '(maximum of: ' + ", ".join(map(str, lowerbounds)) + ')')
        return max(lowerbounds)

    def upperbound(self, upperbound_effort=2):
        # a collection of algorithms that produce an upperbound on OPT, which is usefull for DFS algorithms.

        # Taken from paper  http://www.cs.bris.ac.uk/~montanar/papers/floodit.pdf
        # Theorem 4
        def paper(n,c):
            return 2* n + (2 * c )**0.5 * n + c


        # if we start at 0,0, we can add a complete new row+column by iterating through all colors. Repeat this n-1 times and all is captured
        def mine_simple(n,c):
            return (n-1) * c


        # For every scanline i (row i + column i) we have:
        #   - at most (2*(i+1))-1 positions that needs to be captured.
        #   - at most  c colors (chosen in the right ordering with color of data[i][i] last)
        # so we need min((2*i)-1,c) per scanline i, total = sum_i: min((2*i)-1,c) for 1 <= i < n
        # In fact, this gives rise to the recur algorithm, which can be a little better in practise.
        def mine_better(n,c):
            total = 0
            for i in range(1,n):
                total += min((2*(i+1))-1,c)
            return total

        def majority_merge_all_nodes(self):
            sequences = []
            for i in range(0, self.field_size):
                for j in range(0, self.field_size):
                    # all paths to elements from root
                    sequences.append([self.clustercolor[x] for x in self.shortest_path(self.field_to_cluster[(0, 0)], self.field_to_cluster[(i, j)])])
            scs = ts.majority_merge(ts.remove_redundant_sequences(sequences))
            if self.verbose:
                print("Best MMW move", len(scs), scs)
            return scs

        def scs_backtrack_majorities_depth2(self):
            sequences = []
            for i in range(0, self.field_size):
                for j in range(0, self.field_size):
                    # all paths to elements from root
                    sequences.append([self.clustercolor[x] for x in self.shortest_path(self.field_to_cluster[(0, 0)], self.field_to_cluster[(i, j)])])
            scs = ts.scs_backtrack_majorities([], ts.remove_redundant_sequences(sequences), depth=2)
            if self.verbose:
                print("Best backtrack_2-MM move", len(scs), scs)
            return scs

        ### Field specific initializations

        n = self.field_size
        # c = self.num_colors
        # rather, unique colors (some might not be there)
        c = len([x for x in self.colorsleft if x])


        upperbound = [
                      mine_better(n,c),
                      len(self.clusters)-1,
                      paper(n,c)
                     ]

        if upperbound_effort > 0:
            # find upperbound very quick
            scs = majority_merge_all_nodes(self)
            rec = self.recur(True)
            self.reset()
            be = self.backedges()
            it = self.iterative()
            self.reset()
            it_1 = self.iterative_1()
            self.reset()
            gc = self.greedy_clusters()
            self.reset()

            upperbound.append(len(scs))
            upperbound.append(len(rec))
            upperbound.append(len(be))
            upperbound.append(len(it))
            upperbound.append(len(it_1))
            upperbound.append(len(gc))

            if upperbound_effort > 1:
                # backtrack more deeply regarding majority merge
                bmd2 = scs_backtrack_majorities_depth2(self)

                # more extensive subsetting and randomizing on upperbounds found so far
                run_time_per_simplify = 2
                rm = self.random_moves(run_time_per_simplify) #try random 2 secs
                rm_sub = self.subsets_some(rm, run_time_per_simplify)
                scs_sub = self.subsets_some(scs, run_time_per_simplify)
                rec_sub = self.subsets_some(rec, run_time_per_simplify)
                be_sub = self.subsets_some(be, run_time_per_simplify)
                it_sub = self.subsets_some(it, run_time_per_simplify)
                it_1_sub = self.subsets_some(it_1, run_time_per_simplify)
                

                upperbound.append(len(bmd2))
                upperbound.append(len(rec_sub))
                upperbound.append(len(rm))
                upperbound.append(len(rm_sub))
                upperbound.append(len(scs_sub))
                upperbound.append(len(be_sub))
                upperbound.append(len(it_sub))
                upperbound.append(len(it_1_sub))


        self.upperbound = upperbound
        if self.verbose:
            print("Upperbound returned", min(upperbound), '(minimum of: ' + ", ".join(map(str, upperbound)) + ')')
        return min(upperbound)



    def make_move(self, color):
        if color > self.num_colors or color < 1:
            print("Invalid color")
        else:
            self.current_color = color
            self.move_count += 1
            obtain = set()
            for p in self.posesses_clusters:
                if color in self.ccc[p]:
                    for s in self.ccc[p][color]:
                        if s not in self.posesses_clusters:
                            if s not in obtain:
                                self.colorsleft[color-1] -= 1
                            obtain.add(s)
            for o in obtain:
                self.posesses_clusters.append(o)
                for tup in self.clusters[o][1]:
                    if tup not in self.posesses:
                        self.posesses.append(tup)

        won = self.check_win(self.posesses)
        return self.move_count if won else -1

    def save(self):
        # TODO: method that saves interesting random field in the field-data file.
        pass

    def check_win(self, posesses):
        return len(self.clusters) == len(self.posesses_clusters)

    def obtain_neighbours(self, x, y, colors):
        n = []
        if y < self.field_size -1 and self.data[x][y+1] in colors:
            n.append((x,y+1))
        if x < self.field_size -1 and self.data[x+1][y] in colors:
            n.append((x+1,y))
        if x > 0 and self.data[x-1][y] in colors:
            n.append((x-1,y))
        if y > 0 and self.data[x][y-1] in colors:
            n.append((x,y-1))
        return n

    def recur(self, extended):
        best_move = []
        for d in range(0, self.field_size-1):
            keypoint = (d, d)
            nextkey = (d+1, d+1)
            # make sure we add the next keypoint
            colors = [self.data[d+1][d+1]]
            for i in range(d+1):
                val = self.data[i][d+1]
                if not val in colors:
                    colors.append(val)
                val = self.data[d+1][i]
                if not val in colors:
                    colors.append(val)
            colors.reverse()
            for c in colors:
                best_move.append(c)
        # best_move now has valid solution
        if not extended:
            return best_move

        # So lets simplify it, removing doubles never hurts
        i = 0
        while i < len(best_move)-1:
            if best_move[i] == best_move[i+1]:
                best_move.pop(i)
            i+=1

        # strip the end, cause we might have a valid solution already
        self.reset()
        for i, move in enumerate(best_move):
            if self.make_move(move) >= 0:
                break
        if self.verbose:
            print("Best recur move", len(best_move[:i+1]), best_move[:i+1])

        return best_move[:i+1]

    def cfc(self):
        # computes backward edges cluster-from_color-cluster
        # counterpart of ccc
        cfc = {}
        for x in self.ccc:
            cfc[x] = {}
        for x in self.ccc:
            for color in self.ccc[x]:
                for c in self.ccc[x][color]:
                    try:
                        cfc[c][color].add(x)
                    except Exception as e:
                        cfc[c][color] = set([x])
        return cfc

    def degree(self, cluster):
        return sum(len(self.ccc[cluster][color]) for color in self.ccc[cluster])

    def backedges(self):
        # semi-greedy home-grown approach that decides which color opens most edges to new clusters
        cfc = self.cfc()
        posesset = set(self.posesses_clusters)

        backedges_solution = [self.data[0][0]]
        while len(posesset) < len(self.clusters):
            # determine possible moves
            possible_moves = []
            for p in posesset:
                for color in self.ccc[p]:
                    if color not in possible_moves and color != backedges_solution[-1]:
                        possible_moves.append(color)

            # compute sets gained for each move. (TODO: can be merged into previous loop)
            # also keep track of all_obtainable for weighing
            gain_per_move = dict(list(zip(possible_moves, [set() for x in possible_moves])))
            all_obtainable = set()
            for color in possible_moves:
                obtained = set()
                for p in posesset:
                    if color in self.ccc[p]:
                        for c in self.ccc[p][color]:
                            if c not in posesset:
                                obtained.add(c)
                                all_obtainable.add(c)
                gain_per_move[color] = obtained

            # compute weight function for each choice based on sets gained
            maximum = -1
            for color in gain_per_move:
                if gain_per_move[color]:
                    total = 0.0
                    for c in gain_per_move[color]:
                        # we use the number of outgoing edges from an obtained cluster c to c' where c' not in posessets
                        # total += len([x for x in cfc[c][color] if x not in posesset])
                        for x in cfc[c][color]:
                            if x in posesset:
                                total += 0.1
                            elif x in all_obtainable:
                                total += 0.5
                            else:
                                total += 1.0 * math.log(1+len(self.clusters[x][1]))
                    if total > maximum:
                        next_move = color
                        maximum = total

            if maximum == 0:
                # print "All backedges lead to stuff we already have, just add remaining colors in arbitrary order.."
                for color in gain_per_move:
                    if gain_per_move[color]:
                        backedges_solution.append(color)
                break

            backedges_solution.append(next_move)
            posesset = posesset.union(gain_per_move[next_move])

        backedges_solution = backedges_solution[1:]
        if self.verbose:
            print("Best backedges move", len(backedges_solution), backedges_solution)
        return backedges_solution



    def random_moves(self, run_time):
        run_until = time.time() + run_time
        best_move = []
        best = sys.maxsize
        while time.time() < run_until:
            self.reset()
            move_trace = []
            next_move = self.current_color
            won = -1
            while won < 0 and len(move_trace) < best:
                last_move = next_move
                # all colors = [1,2,3,4]
                # f = random (2,4) -> (2,3,4)
                # if last = 1 -> ok
                # if last = 2 -> (1,3,4), so 2 == 1
                next_move = random.randint(2, self.num_colors)
                if next_move == last_move:
                    next_move = 1
                assert(next_move != last_move)
                won = self.make_move(next_move)
                move_trace.append(next_move)

            if won != -1 and won < best:
                best_move = move_trace
                best = won
        if self.verbose:
            print("Best random move", len(best_move), best_move)
        return best_move

    def solve(self, solution, display_steps=True, flush=False):
        self.reset()
        win = False
        for color in solution:
            self.make_move(color)
            if display_steps:
                time.sleep(0.3)
                if flush:
                    print("\n" * 2)
                    os.system(['clear','cls'][os.name == 'nt'])
                data_copy = []
                for i in self.data:
                    data_copy.append(copy.copy(i))
                for (x,y) in self.posesses:
                    data_copy[x][y] = " "
                print("Last move = %i, movecount = %i"%(self.current_color, self.move_count))
                for i in data_copy:
                    for j in i:
                        print(j, end=' ')
                    print()                
            win = self.check_win(self.posesses)
            if win:
                break
        if display_steps:
            print("Won =", win)
        return win


    def subsets_rand(self, valid_move, result, until):
        size = len(valid_move)-1
        if until < time.time():
            return result
        for x in range(len(valid_move) // 4):
            # try removing a random number
            i = random.randint(0,size)
            removed = valid_move.pop(i)
            if ((not size in result) or not valid_move in result[size]) and self.solve(valid_move, False):
                if size in result:
                    result[size].append(copy.copy(valid_move))
                else:
                    result[size] = [copy.copy(valid_move)]
                self.subsets_rand(valid_move, result, until)
            valid_move.insert(i, removed)
        return result


    def subsets_some(self, valid_move, run_time):
        if not self.solve(valid_move, False):
            print("Supplied solution was not a valid solution:")
            print("#", len(valid_move), valid_move)
        else:
            result = self.subsets_rand(valid_move, {}, time.time()+run_time)
            if not result:
                if self.verbose:
                    print("Same subsets move", len(valid_move), valid_move)
                return valid_move
            else:
                total = 0
                for key in sorted(result, reverse=True):
                    total += len(result[key])
                best_move = result[key][0]
                if self.verbose:
                    if len(best_move) == len(valid_move):
                        print("Same subsets move", len(best_move), best_move)
                    else:
                        print("Best subsets move", len(best_move), best_move)
                return best_move

    def iterative(self):
        self.reset()
        next_move = self.num_colors
        move_trace = []
        won = -1
        prev_posesses_clusters = 1
        while won < 0:
            last_move = next_move
            next_move = (last_move ) % self.num_colors + 1
            won = self.make_move(next_move)
            if len(self.posesses_clusters) > prev_posesses_clusters:
                prev_posesses_clusters = len(self.posesses_clusters)
                move_trace.append(next_move)
        if self.verbose:
            print("Best iterative move", len(move_trace), move_trace)
        return move_trace

    def iterative_1(self):
        self.reset()
        increasing = True
        next_move = min(self.current_color, self.num_colors-1)
        move_trace = []
        won = -1
        prev_posesses_clusters = 1
        while won < 0:
            last_move = next_move
            if increasing:
                next_move = last_move + 1
                if next_move == self.num_colors:
                    increasing = False
            else:
                next_move = last_move - 1
                if next_move == 1:
                    increasing = True

            won = self.make_move(next_move)
            if len(self.posesses_clusters) > prev_posesses_clusters:
                prev_posesses_clusters = len(self.posesses_clusters)
                move_trace.append(next_move)
        if self.verbose:
            print("Best iterative_1 move", len(move_trace), move_trace)
        return move_trace

    def greedy_clusters(self):
        choices = [self.data[0][0]]
        while len(self.posesses_clusters) != len(self.clusters):
            possible_color_choices = [x+1 for x in range(self.num_colors) if x+1 != choices[-1]]
            choose_color = possible_color_choices[0] #if tied for greediness, first is chosen
            choose_sum = -1
            for color in possible_color_choices:    #determine which color obtains most clusters
                would_obtain = set()
                for p_cluster in self.posesses_clusters:
                    if color in self.ccc[p_cluster]:
                        for c in self.ccc[p_cluster][color]:
                            if c not in self.posesses_clusters:
                                would_obtain.add(c)
                if would_obtain:
                    totalsum = 0
                    for cluster in would_obtain:
                        subsum = sum(len(self.ccc[cluster][a_color]) for a_color in self.ccc[cluster])
                        totalsum += subsum
                    if totalsum > choose_sum:
                        choose_sum = totalsum
                        choose_color = color
            choices.append(choose_color)
            self.make_move(choose_color)
        choices = choices[1:]
        if self.verbose:
            print("Best greedy_clusters move", len(choices), choices)
        return choices

    def shortest_path(self, start, end):
        D = {}	    # dictionary of final distances
        P = {}	    # dictionary of predecessors
        Q = {}      # estimated distances of non-final cluster.
        Q[start] = 0
        visited = [start]
        for i in range(len(self.clusters)-1):
            v = visited[i]
            D[v] = Q[v]
            for color in self.ccc[v]:
                for w in self.ccc[v][color]:
                    vwLength = D[v] + 1
                    if w in D:
                        if vwLength < D[w]:
                            raise ValueError
                    elif w not in Q or vwLength < Q[w]:
                        Q[w] = vwLength
                        visited.append(w)
                        P[w] = v

        path = []
        while end != start:
            path.append(end)
            end = P[end]
        path.reverse()
        return path

    def dfs(self, posessets, turns, choices):
        solutions = []
        # prune away solutions where (turns + colors_left_in_field) >= best so far
        # also means that if the upperbound is already correct, no solution will be found..
        if turns + len([x for x in self.colorsleft if x > 0]) - 1 < self.upper_bound:
            lenpos = len(posessets)
            obtain = [posessets.copy() for color in range(self.num_colors)]
            # compute possible moves (colors) and find the sets we obtain when making that move.
            possible_moves = []
            for p in posessets:
                for color in self.ccc[p]:
                    if color != choices[-1]:
                        for q in self.ccc[p][color]:
                            if q not in obtain[color-1]:
                                obtain[color-1].add(q)
                                if color-1 not in possible_moves:
                                    possible_moves.append(color-1)
            for color in possible_moves:
                # check if there is a color that can be chosen such that all occurences of this color are gone.
                # this is optimal and faster as we dont need to evaluate it again for rest of recursion.
                if (self.colorsleft[color] - (len(obtain[color]) - lenpos)) == 0:
                    self.colorsleft[color] = 0
                    choices.append(color+1)
                    if len(obtain[color]) == self.num_clusters:
                        solutions.append((turns, choices[1:]))
                        self.upper_bound = turns
                        print("#", turns, choices[1:])
                    else:
                        for x in self.dfs(obtain[color], turns+1, choices):
                            solutions.append(x)
                    choices.pop()
                    self.colorsleft[color] += (len(obtain[color]) - lenpos)
                    break
            else:
                for color in possible_moves:
                    # note color + 1 is the actual color
                    if not self.cache.cached(turns, obtain[color]):
                        self.colorsleft[color] -= (len(obtain[color]) - lenpos)
                        for x in self.dfs(obtain[color], turns+1, choices+[color+1]):
                            solutions.append(x)
                        self.colorsleft[color] += (len(obtain[color]) - lenpos)
        return solutions


    def bfs(self, posessets, turns, choices):
        solutions = []
        Q = []
        Q.append((self.colorsleft[:], choices, posessets, set()))
        while Q:
            colorsleft, choices, posessets, contained = Q.pop(0)
            turns = len(choices)
            if turns + len([x for x in colorsleft if x > 0]) -1 < self.upper_bound:
                obtain = [posessets.copy() for color in range(self.num_colors)]
                # TODO: copy is expensive, how about shrinking with contained?
                possible_moves = []
                for p in posessets:
                    if p not in contained:
                        obtained_new = False
                        for color in self.ccc[p]:
                            if color != choices[-1]:
                                for q in self.ccc[p][color]:
                                    if q not in obtain[color-1] and q not in contained:
                                        obtained_new = True
                                        obtain[color-1].add(q)
                                        if color-1 not in possible_moves:
                                            possible_moves.append(color-1)
                        if not obtained_new:
                            contained.add(p)
                for color in possible_moves:
                    # check if there is a color that can be chosen such that all occurences of this color are gone.
                    # this is optimal and faster as we dont need to evaluate it again for rest of recursion.
                    if (colorsleft[color] - (len(obtain[color]) - len(posessets))) == 0:
                        cl = colorsleft[:]
                        cl[color] = 0
                        choices.append(color+1)
                        if (len(obtain[color])) == self.num_clusters:
                            solutions.append((turns, choices[1:]))
                            self.upper_bound = turns
                            print('#', turns, choices[1:])
                        else:
                            Q.append((cl, choices[:], obtain[color], contained.copy()))
                        break
                else:
                    # try all possible colors
                    for color in possible_moves:
                        # note color + 1 is the actual color
                        if not self.cache.cached(turns, obtain[color]):
                            if not random.randint(0,100000):
                                print("Q size", len(Q), "-", len(choices), choices[1:]+[color+1])
                                self.cache.print_cache_stats()
                                self.print_clusterdata(obtain[color])
                            cl = colorsleft[:]
                            cl[color] -= (len(obtain[color]) - len(posessets))
                            Q.append((cl, choices[:]+[color+1], obtain[color], contained.copy()))
        self.cache.print_cache_stats()
        return solutions


    def bfs_genetic(self, posessets, turns, choices):
        # this is like BFS, but now at depth x, we only keep the 'best' n intermediate solutions according to a magic metric.

        def metric1(queue_entry):
            (colorsleft, choices, posessets) = queue_entry 
            return 1.0/(len(posessets))

        def metric2(queue_entry):
            (colorsleft, choices, posessets) = queue_entry 
            return sum(colorsleft) \
                   + 1.0/(len(posessets)) \
                   + len([e for e in colorsleft if e != 0])

        keep_best = 500     # increase number for better accuracy at the cost of running time
        solutions = []
        Q = []
        Q.append((self.colorsleft[:], choices, posessets))
        current_depth = 1
        while Q:
            turns = len(Q[0][1])
            if turns > current_depth:
                if len(Q) > keep_best:
                    print("Throwing away", len(Q) - keep_best, "out of", len(Q), "at depth", current_depth)
                    Q = sorted(Q, key=metric2)[:keep_best]

                current_depth = turns
            colorsleft, choices, posessets = Q.pop(0)
            if turns + len([x for x in colorsleft if x > 0]) -1 < self.upper_bound:
                obtain = [posessets.copy() for color in range(self.num_colors)]
                possible_moves = []
                for p in posessets:
                    for color in self.ccc[p]:
                        if color != choices[-1]:
                            for s in self.ccc[p][color]:
                                if s not in obtain[color-1]:
                                    obtain[color-1].add(s)
                                    if color-1 not in possible_moves:
                                        possible_moves.append(color-1)
                for color in possible_moves:
                    # check if there is a color that can be chosen such that all occurences of this color are gone.
                    # this is optimal and faster as we dont need to evaluate it again for rest of recursion.
                    if (colorsleft[color] - (len(obtain[color]) - len(posessets))) == 0:
                        cl = colorsleft[:]
                        cl[color] = 0
                        choices.append(color+1)
                        if len(obtain[color]) == self.num_clusters:
                            solutions.append((turns, choices[1:]))
                            self.upper_bound = turns
                        else:
                            Q.append((cl, choices[:], obtain[color]))
                        break
                else:
                    # try all possible moves
                    for color in possible_moves:
                        if not self.cache.cached(turns, obtain[color]):
                            cl = colorsleft[:]
                            cl[color] -= (len(obtain[color]) - len(posessets))
                            Q.append((cl, choices[:]+[color+1], obtain[color]))
        return solutions
     
