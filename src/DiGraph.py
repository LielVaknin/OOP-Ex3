import copy
from src.node_data import NodeData


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
        Returns the number of exist nodes in this graph.
        @return: The number of exist nodes in this graph.
        """
        return self.__node_size

    def e_size(self) -> int:
        """
        Returns the number of exist edges in this graph.
        @return: The number of exist edges in this graph.
        """
        return self.__edge_size

    def get_all_v(self) -> dict:
        """
        Returns a dictionary of all the nodes in the graph, each node is represented using a pair.
         (key: node_id, value: node_data).
        """
        return self.__nodes

    def all_in_edges_of_node(self, id1: int) -> dict:
        """
        Returns a dictionary of all the edges connected to (into) node_id,
        each edge is represented using a pair (key: other_node_id, value: weight of edge).
         """
        ans = {}
        if id1 not in self.__nodes:
            return ans
        return self.end_edges[id1]

    def all_out_edges_of_node(self, id1: int) -> dict:
        """
        Returns a dictionary of all the edges connected from (out) node_id, each edge is represented using a pair
        (key: other_node_id, value: weight of edge).
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
        @param id1: Represents the src node of the edge.
        @param id2: Represents the dest node of the edge.
        @param weight: Represents the weight of the edge.
        @return: This method returns True if the edge was added successfully, False o.w.
        If the edge already exists or one of the nodes dose not exists the function returns False.
        """
        if id1 not in self.__nodes or id2 not in self.__nodes or id1 == id2 or weight < 0 or id2 in self.edges[id1]:
            return False
        self.edges[id1][id2] = weight
        self.end_edges[id2][id1] = weight
        self.__mc += 1
        self.__edge_size += 1
        return True

    def add_node(self, node_id: int, pos: tuple = None) -> bool:
        """
        Adds a node to the graph.
        @param node_id: Represents the key of the node.
        @param pos: Represents the position (3D point) of the node.
        @return: True if the node was added successfully, False o.w.
        If the node id already exists the node will not be added and the method will return False.
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
        This method removes all the edges comes out and in of node_id and finally removes node_id from the graph.
        @param node_id: Represents the key of the node.
        @return: True if the node was removed successfully, False o.w.
        If the node id does not exists the function will return False.
        """
        if node_id not in self.__nodes:
            return False

        out_edges = self.all_out_edges_of_node(node_id)
        for n in out_edges:
            del self.end_edges[n][node_id]
            self.__mc += 1
            self.__edge_size -= 1
        del self.edges[node_id]
        in_edges = self.all_in_edges_of_node(node_id)
        for n in in_edges:
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
        @param node_id1: Represents The src node of the edge.
        @param node_id2: Represents The dest node of the edge.
        @return: True if the edge was removed successfully, False o.w.
        If such an edge does not exists the function will return False.
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

    def __str__(self):
        return f'nodes:{self.__nodes.keys()} edges : {self.edges} end edges:{self.end_edges}'

    def __copy__(self):
        new_grpah = copy.deepcopy(self)
        return new_grpah
