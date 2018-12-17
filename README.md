# Flood-it
This repository contains Algorithms, Complexity Analysis and Solvers for the FloodIt game. 

## Purpose
Perhaps you may have played one of these variants of the combinatorial game called Flood-it:
- lemoda.net:  https://www.lemoda.net/javascript/flood-it/index.html
- color-flow:  http://color-flow.appspot.com
- android: https://play.google.com/store/apps/details?id=com.labpixies.flood
- ios: https://itunes.apple.com/us/app/flood-it/id476943146

Being intrigued by optimizing the minimum number of moves, it was time to fire up some code. Despite having perfect information, solving an *n* x *n* board with *c* colours gets complex rather quickly with large values for *n* and/or *c*. The problem can be reduced to the Shortest Common Supersequence [SCS](https://en.wikipedia.org/wiki/Shortest_common_supersequence_problem), meaning no polynomial time algorithm exists, unless *P=NP*. However, several approximation algorithms can be devised. The code contains usages of SCS approximation algorithms to find good upperbounds on the optimal solutions. A Genetic algorithm has been built which does great in general, although some anti-cases have been produced. Also, both Depth First Search and Breadth First Search algorithms on Graphs are in place to always produce correct solutions, albeit at the cost of running time for large *n*.  

## Running it yourself
Run solver.py with Python 3. Choose a mode you like, i.e. interactive, or let some of the various algorithms solve the game for you. Add your own favorite starting state in the field_data file.

## Related documentation
- https://arxiv.org/abs/1001.4420
- https://kunigami.blog/2012/09/16/flood-it-an-exact-approach/
- https://people.maths.ox.ac.uk/scott/Papers/floodit.pdf
