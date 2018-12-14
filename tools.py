import random
import collections
import itertools
import sys

def supersequence(x, y):
    # True if and only if all elements in y occur in x in order (x is a supersequence of y)
    idx = 0
    try:
        for i in y:
            idx = x.index(i, idx)+1
    except ValueError as e:
        return False
    else:
        return True


def subsequence(x, y):
    # x is a subsequence of y  <==>  y is a supersequence of x
    return supersequence(y, x)


def is_scs(scs, sequences):
    result = True
    for s in sequences:
        if not supersequence(scs, s):
            print(scs, "NO supersequence of", s)
            result = False
        else:
            print(scs, "is supersequence of", s)
    return result


def remove_redundant_sequences(redundant_sequences, debug=False):
    if debug:
        for i, rs in enumerate(redundant_sequences):
            print("RS", i,rs)

    count = 0
    # remove doublures
    sequences = []
    for i,s in enumerate(redundant_sequences):
        remove = False
        for j in range(i+1, len(redundant_sequences)):
            if s == redundant_sequences[j]:
                remove = True
                if debug:
                    print(i,j, "DD", s, redundant_sequences[j])
        if remove:
            count +=1
        else:
            sequences.append(s)

    # remove supersequences
    sequences_p = []
    for i,s in enumerate(sequences):
        found = False
        for j,s2 in enumerate(sequences):
            if ((i != j) and (supersequence(s2, s))):
                #print "redundant", s
                count += 1
                found = True
                if debug:
                    print(i,j, "SS", s2, "-", s)
                break
        if not found:
            sequences_p.append(s)
    if debug:
        print("Redundant sequences removed", count)
    return sequences_p


def scs_length(s1, s2):
    S = []
    for i in range(0, len(s1)+1):
        S.append([-1]*(len(s2)+1))
    for i in range(len(s1)+1):
        S[i][0] = i
    for j in range(len(s2)+1):
        S[0][j] = j
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            if s1[i-1] == s2[j-1]:
                S[i][j] = min(S[i-1][j]+1, S[i][j-1]+1, S[i-1][j-1]+1)
            else:
                S[i][j] = min(S[i-1][j]+1, S[i][j-1]+1)
    return S[len(s1)][len(s2)]


def alphabet_leftmost(sequences, permutation):
    scs = []
    i = 0
    while any(sequences):
        found = False
        for s in sequences:
            if s and s[0] == permutation[i]:
                found = True
        if found:
            for s in sequences:
                if s and s[0] == permutation[i]:
                    s.remove(permutation[i])
            scs.append(permutation[i])
        i = (i+1) % len(permutation)
    return scs

def alphabet_leftmost_rand(sequences, num_colors):
    scs = []
    while any(sequences):
        next = random.randint(1, num_colors)
        found = False
        for s in sequences:
            if s and s[0] == next:
                found = True
        if found:
            for s in sequences:
                if s and s[0] == next:
                    s.remove(next)
            scs.append(next)
    return scs


def scs_backtrack(moves, sequences, best=100000):
    solution = []
    if any(sequences) and len(moves) < best:
        firsts = set([s[0] for s in sequences if s])
        for c in firsts:
            indices = []
            # remove c 
            for i,s in enumerate(sequences):
                if s and s[0] == c:
                    s.remove(c)
                    indices.append(i)
            a_solution = scs_backtrack(moves+[c], sequences, best)
            if a_solution and len(a_solution) < best:
                solution = a_solution[:]
                best = len(a_solution)
            # restore c
            for i in indices:
                sequences[i] = [c]+sequences[i]
    else:
        if random.random() < 0.00001:
            print("#", len(moves), moves)
        solution = moves[:]

    return solution

def majority_merge(sequences):
    scs = []
    while any(sequences):
        # firsts contains list of first item of a sequence and it's weight (the list's length)
        firsts = [(s[0], len(s)) for s in sequences if s]
        counts = {}
        for x in firsts:
            if x[0] in counts:
                counts[x[0]] += x[1]    #use +1 if not weighted MM
            else:
                counts[x[0]] = x[1]     #use 1 if not weighted MM
        most_common = sorted(counts, key=lambda x: counts[x], reverse=True)[0]
        scs.append(most_common)
        # print "-"*40, most_common, "-"*40
        for s in sequences:
            if s and s[0] == most_common:
                s.remove(most_common)
    return scs

# seems to perform better in standalone method rather than passing self along
def obtain_neighbours_single_color_self_free(data, field_size, x, y, color):
    result = []
    if y < field_size -1 and data[x][y+1] == color:
        result.append((x,y+1))
    if x < field_size -1 and data[x+1][y] == color:
        result.append((x+1,y))
    if x > 0 and data[x-1][y] == color:
        result.append((x-1,y))
    if y > 0 and data[x][y-1] == color:
        result.append((x,y-1))
    return result


def matrix_to_field_rep(m):
    r = []
    for a in m:
        for b in a:
            r.append(b)
    return r

def hashOfSet(s):
    h = str(sorted(s))
    return h


def inttocolor(i):
    if i == 6:
        return "purple"
    elif i == 1:
        return "blue"
    elif i == 2:
        return "green"
    elif i == 3:
        return "red"
    elif i == 4:
        return "pink"
    elif i == 5:
        return "orange"
    elif i == 7:
        return "beige"
    elif i == 8:
        return "cyan"
    elif i == 9:
        return "gold"
    elif i == 10:
        return "skyblue"
    elif i == 11:
        return "lightpink"
    elif i == 12:
        return "peru"
    elif i == 13:
        return "coral2"
    elif i == 14:
        return "blueviolet"
    elif i == 15:
        return "turquoise"
    elif i == 16:
        return "gray"
    else:
        return "black"

#diamonds theory
def paper2(n,c):
    if c > n**2:
        return 0
    else:
        return ((c-1)**0.5 * n / 2) - (c / 2)

def base10toN(num,n):
    new_num_string = ''
    current = num
    while current != 0:
        remainder = current%n
        remainder_string = str(remainder)
        new_num_string = remainder_string+new_num_string
        current = current/n
    return new_num_string

def add_b(n,base):
    i = len(n)-1
    n[i] += 1
    while n[i] > base:
        n[i] = 1
        i -= 1
        n[i] += 1
    return n

def all_unique_hashes(prefix, num_colors, boardsize):
    # generator version, no need to keep all hashes
    if len(prefix) == boardsize:
        yield prefix
    else:
        for i in range(1,min(max(prefix)+1,num_colors)+1):
            for h in all_unique_hashes(prefix+[i], num_colors, boardsize):
                yield h

def count_all_unique_hashes(prefix, num_colors, boardsize):
    if len(prefix) == boardsize:
        return 1
    else:
        s = 0
        for i in range(1,min(max(prefix)+1,num_colors)+1):
            s += count_all_unique_hashes(prefix+[i], num_colors, boardsize)
        return s

def list_to_int(l,base):
    result = 0
    for i,x in enumerate(reversed(l)):
        result += (x-1)*(base**i)
    return result

def print_field_rep(field_rep, field_size):
    if not field_rep:
        print("   tight lowerbound")
    else:
        for x in range(field_size):
            print("  ", field_rep[x*field_size:(x+1)*field_size])

def int_to_base_as_list(x,b,length):
    rets = []
    while x>0:
        x,idx = divmod(x,b)
        rets = [idx+1] + rets
    rets = [1]*(length - len(rets))+rets
    return rets

def field_rep_is_simple(field_rep, num_colors, field_size):
    # function for finding fields that do not belong to the toughest fields (e.g. easy fields), returns True if so.

    # not all colors used, or at least all colors are used when colors > fieldsize**2
    if len(set(field_rep)) < min(num_colors, field_size**2):
        return True

    # 4 consecutive horizontal (linebreak overlapping also)
    consecutive_4 = 1
    for i in range(len(field_rep)-1):
        if field_rep[i] == field_rep[i+1]:
            consecutive_4 += 1
            if consecutive_4 == 4:
                break
        else:
            consecutive_4 = 1
    if consecutive_4 == 4:
        return True

    # 3 consecutive vertical
    for i in range(field_size-3):
        consecutive_3 = 1
        val = field_rep[i]
        for j in range(1,4):
            if field_rep[i+j*field_size] == val:
                consecutive_3 += 1
            else:
                break
        if consecutive_3 == 3:
            return True

    # 4 block
    for i in range(field_size-1):
        for j in range(field_size-1):
            if field_rep[i+j*field_size] == field_rep[i+1+j*field_size] and field_rep[i+j*field_size] == field_rep[i+1+(j+1)*field_size] and field_rep[i+j*field_size] == field_rep[i+1+j*field_size]:
                return True
    return False


def pretty_print_scs_with_sequences(scs, sequences):
    print("SCS: " + "".join([str(s) for s in scs]))
    for i,aseq in enumerate(sequences):
        seq = [-1]*len(scs)
        idx = 0
        for v in aseq:
            idx = scs.index(v, idx)+1
            seq[idx-1] = v
        print((2-len(str(i)))*" "+"s%s: "%i + "".join([str(s) for s in seq]).replace("-1", "."))


# Backtracking shortest common supersequence of sequences
# has a branching on the leftmost 'depth' most common
# With depth = 1, it is the same as the weighted majority merge  algorithm
# With depth = infinity, it will backtracks all possibilities
def scs_backtrack_majorities(moves, sequences, best=sys.maxsize, depth=1):
    solution = []
    if any(sequences) and len(moves) + len(set(itertools.chain.from_iterable(sequences))) < best:
        firsts = [(s[0], len(s)) for s in sequences if s]
        counts = {}
        for x in firsts:
            if x[0] in counts:
                counts[x[0]] += x[1]    #use +1 if not weighted MM
            else:
                counts[x[0]] = x[1]     #use 1 if not weighted MM
        most_common = sorted(counts, key=lambda x: counts[x], reverse=True)[:depth]
        
        for c in most_common:
            indices = []
            # remove c 
            for i,s in enumerate(sequences):
                if s and s[0] == c:
                    s.remove(c)
                    indices.append(i)
            a_solution = scs_backtrack_majorities(moves+[c], sequences, best, depth)
            if a_solution and len(a_solution) < best:
                solution = a_solution[:]
                # print solution
                best = len(a_solution)
            # restore c
            for i in indices:
                sequences[i] = [c]+sequences[i]
    else:
        if random.random() < 0.000001:
            print("intermediate #", len(moves), moves, "best found so far", best)
        if not any(sequences):
            solution = moves[:]
            best = len(solution)

    return solution




if __name__ == "__main__":
    # some tests on the algorithms here

    best = 100000000
    # list_of_sequences = [[2], [2, 1], [2, 1], [2, 1, 4], [2, 1, 4, 1], [2], [2, 3], [2, 1, 2], [2, 1, 4], [2, 1, 4, 5], [2, 1, 4, 5, 4], [2, 1], [2, 3, 2], [2, 1, 2, 1], [2, 1, 2, 1], [2, 1, 2, 1, 4], [2, 1, 2, 1, 3, 1], [2, 1], [2, 1, 5], [2, 1, 2, 1], [2, 1, 2, 1], [2, 1, 2, 1, 3], [2, 1, 2, 1, 3, 1], [2, 1, 5], [2, 1, 5, 6], [2, 1, 2, 1, 5], [2, 1, 2, 1, 3], [2, 1, 2, 1, 3, 2], [2, 1, 2, 1, 3, 2, 3], [2, 1, 5, 1], [2, 1, 5, 6, 5], [2, 1, 2, 1, 3, 1], [2, 1, 2, 1, 3, 1], [2, 1, 2, 1, 3, 1, 3], [2, 1, 2, 1, 3, 1, 3, 1]]
    # list_of_sequences = [[3, 1, 5], [2, 5, 4, 3], [2, 4, 1, 3, 2], [2, 4, 1, 2, 4, 3]]
    # non_opt_scs = [1, 3, 1, 5, 6, 3, 4, 1, 5, 4, 2, 6, 1, 3, 4, 6, 1, 2, 5, 6, 1, 4, 5]
    # list_of_sequences = [ [4, 5, 6, 3, 2, 4],[1, 5, 2, 4, 6, 1, 2],[1, 3, 6, 3, 1, 5, 3, 1],[1, 3, 1, 5, 6, 5, 6, 1, 6],[1, 3, 1, 5, 1, 4, 2, 5, 6, 5],[1, 3, 1, 6, 3, 5, 4, 1, 2, 1],[1, 3, 1, 6, 3, 4, 1, 5, 6, 2]]
    list_of_sequences = [[1, 4, 1, 3, 1, 3, 6],[1, 4, 1, 3, 1, 2, 6, 2],[1, 4, 1, 3, 1, 2, 1, 5, 4],[1, 4, 2, 4, 6, 3, 2, 1, 2],[5, 6, 2, 4],[1, 4, 3, 5, 3, 1, 4, 3, 6, 4],[1, 4, 3, 5, 3, 1, 4, 3, 1, 4, 5],[1, 4, 3, 5, 2, 5, 3, 5, 3, 5, 1, 2],[1, 4, 2, 4, 6, 3, 2, 5, 3, 5, 3, 6],[1, 4, 2, 4, 6, 3, 2, 5, 3, 5, 3, 4, 2],[1, 4, 3, 5, 2, 5, 3, 1, 5, 6, 5, 3, 5, 4],[1, 4, 3, 5, 2, 4, 5, 6, 3],[1, 4, 3, 5, 2, 4, 5, 6, 4],[1, 4, 3, 5, 2, 4, 5, 6, 2, 5],[1, 4, 3, 5, 2, 5, 3, 1, 5, 2],[1, 4, 3, 5, 2, 5, 3, 1, 5, 4, 3, 6, 3, 1]]
    print(len(list_of_sequences), "sequences")
    simplified = remove_redundant_sequences(list_of_sequences)
    for s in simplified:
        print(s)
    print(len(simplified), "simplified")
    print("---")
    import copy
    simpl = copy.deepcopy(simplified)
    # print len(majority_merge(simplified[:]))
    scs = majority_merge(simplified)
    print(len(scs), scs)
    scs = scs_backtrack_majorities([], simpl, depth=2)
    print(len(scs), scs)
    pretty_print_scs_with_sequences(scs, simpl)
    # print scs_backtrack([], simplified)
    # raise
