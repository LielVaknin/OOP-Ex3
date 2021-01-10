import copy

from node_data import NodeData


class DiGraph:
    """This class represents a directed weighted graph."""

    def __init__(self):
        self.__node_size = 0
        self.__edge_size = 0
        self.__mc = 0
        self.__nodes = {}
        self.edges = {}
        self.end_edges = {}

    def v_size(self) -> int:
        """
        Returns the number of vertices in this graph.
        @return: The number of vertices in this graph.
        """
        return self.__node_size

    def e_size(self) -> int:
        """
        Returns the number of edges in this graph.
        @return: The number of edges in this graph.
        """
        return self.__edge_size

    def get_all_v(self) -> dict:
        """
        Returns a dictionary of all the nodes in the graph, each node is represented using a pair.
         (node_id, node_data)
        """
        return self.__nodes

    def all_in_edges_of_node(self, id1: int) -> dict:
        """
        Returns a dictionary of all the nodes connected to (into) node_id ,
        each node is represented using a pair (other_node_id, weight).
         """
        ans = {}
        if id1 not in self.__nodes:
            return ans
        return self.end_edges[id1]

    def all_out_edges_of_node(self, id1: int) -> dict:
        """
        Returns a dictionary of all the nodes connected from node_id, each node is represented using a pair
        (other_node_id, weight)
        """
        ans = {}
        if id1 not in self.__nodes:
            return ans
        return self.edges[id1]

    def get_mc(self) -> int:
        """
        Returns the current version of this graph,
        on every change in the graph state - the MC should be increased.
        @return: The current version of this graph.
        """
        return self.__mc

    def add_edge(self, id1: int, id2: int, weight: float) -> bool:
        """
        Adds an edge to the graph.
        @param id1: The start node of the edge.
        @param id2: The end node of the edge.
        @param weight: The weight of the edge.
        @return: True if the edge was added successfully, False o.w.
        If the edge already exists or one of the nodes dose not exists the functions will do nothing.
        """
        if id1 not in self.__nodes or id2 not in self.__nodes or id1 == id2 or weight < 0 or id2 in self.edges.get(
                id1):
            return False
        self.edges[id1][id2] = weight
        self.end_edges[id2][id1] = weight
        self.__mc += 1
        self.__edge_size += 1
        return True

    def add_node(self, node_id: int, pos: tuple = None) -> bool:
        """
        Adds a node to the graph.
        @param node_id: The node ID.
        @param pos: The position of the node.
        @return: True if the node was added successfully, False o.w.
        If the node id already exists the node will not be added.
        """
        if node_id in self.__nodes:
            return False
        node = NodeData(node_id, pos)
        self.__nodes[node_id] = node
        self.edges[node_id] = {}
        self.end_edges[node_id] = {}
        self.__mc += 1
        self.__node_size += 1
        return True

    def remove_node(self, node_id: int) -> bool:
        """
        Removes a node from the graph.
        @param node_id: The node ID.
        @return: True if the node was removed successfully, False o.w.
        If the node id does not exists the function will do nothing.
        """
        if node_id not in self.__nodes:
            return False

        out_edges = self.all_out_edges_of_node(node_id)
        for n in out_edges.keys():
            del self.end_edges[n][node_id]
            self.__mc += 1
            self.__edge_size -= 1
        del self.edges[node_id]
        in_edges = self.all_in_edges_of_node(node_id)
        for n in in_edges.keys():
            del self.edges[n][node_id]
            self.__mc += 1
            self.__edge_size -= 1
        del self.end_edges[node_id]
        del self.__nodes[node_id]
        self.__node_size -= 1
        self.__mc += 1
        return True

    def remove_edge(self, node_id1: int, node_id2: int) -> bool:
        """
        Removes an edge from the graph.
        @param node_id1: The start node of the edge.
        @param node_id2: The end node of the edge.
        @return: True if the edge was removed successfully, False o.w.
        If such an edge does not exists the function will do nothing.
        """
        if node_id1 not in self.__nodes or node_id2 not in self.__nodes or node_id1 == node_id2:
            return False
        if node_id2 not in self.edges[node_id1]:
            return False
        del self.edges[node_id1][node_id2]
        del self.end_edges[node_id2][node_id1]
        self.__mc += 1
        self.__edge_size -= 1
        return True

    def set_edges(self, edges_dict, end_edges_dict):
        self.edges = edges_dict
        self.end_edges = end_edges_dict

    def __str__(self):
        return f'nodes:{self.__nodes.keys()} edges : {self.edges} end edges:{self.end_edges}'

    def __copy__(self):
        new_graph = DiGraph
        new_graph.__ = copy.deepcopy(self.__nodes)
        edges = copy.deepcopy(self.edges)
        end_edges = copy.deepcopy(self.end_edges)
        node_size = self.__node_size
        edge_size = self.__edge_size
        mc = self.__mc
