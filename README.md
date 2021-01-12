# A directed weighted graph 
## Decription
### Authors : Liel Vaknin & Yair Aviv

__This project consists of 3 parts:__<br />
__Part A:__ Implements a data structure type of a directed weighted graph in Python.<br />
__Part B:__ Implements algorithms for use on a directed weighted graph in Python.<br />
__Part C:__ Makes a comparison for the implementation of selected algorithms of the graph in Python versus 2 other implementations:<br />
* The first is of ex2 project which implemented in Java.
* The second is of the Python library for graphs - [NetworkX](https://en.wikipedia.org/wiki/NetworkX).
---
## src 
This package contains:<br />
3 python files which each file contains a class.<br />
2 abstract classes which represents interfaces.

* *NodeData class* implemented in the node_data file.<br />
It represents the set of operations applicable on a node in a directed weighted graph.<br />
This class implements the methods:<br /> 
set & get methods, _ _ *init* _ _ method, _ _ *str* _ _ method and _ _ *copy* _ _ method.

* *DiGraph class* implemented in DiGraph file using the definition of GraphInterface.<br /> 
It represents a directed weighted graph.<br /> 
This class implements the methods:<br /> 
_ _ *init* _ _ method, methods for returning the number of nodes / edges in the graph,<br /> 
methods for returning a dictionary of all nodes in the graph / all nodes connected to (into) a given node / all nodes connected from a given node, a method for returning the mc (mode count - counts changes in the graph), methods for adding / removing nodes and edges to / from the graph, _ _ *str* _ _ method and _ _ *copy* _ _ method.

* *GraphAlgo class* implemented in GraphAlgo file. It inherits from the given GraphAlgoInterface abstract class and represents an Undirected (positive) Weighted Graph Theory algorithms.<br />
The class includes a set of operations applicable on a graph type of DiGraph:<br />
_ _ *init* _ _ method which initializes a graph with a given graph, a get_graph method, a method which saves self graph to a given file name and a method which loads a graph to self graph (using [JSON](https://en.wikipedia.org/wiki/JSON) format), a method for finding the shortest path in the graph between a given source and destination and finding its length - using [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm), a method for finding the Strongly Connected Component that a specific node is part of, a method for finding all the Strongly Connected Component in the graph and a method for plotting the graph using [Matplotlib library](https://en.wikipedia.org/wiki/Matplotlib).

## Tests 
This ptoject includes 2 unittest tests :
 -  TestDiGraph - for testing DiGraph class's methods.
 -  TestGraphAlgo - for testing GraphAlgo class's algorithms.

### An example of a directed weighted graph:
![An example of graph](https://github.com/LielVaknin/OOP-Ex3/blob/master/resources/Graph%20example.png)


*For more information go to the project's [wiki pages](https://github.com/LielVaknin/OOP-Ex3/wiki)*


