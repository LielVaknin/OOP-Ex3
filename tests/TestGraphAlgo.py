import copy
import time
import unittest
from unittest import TestCase
from src.DiGraph import DiGraph
from src.GraphAlgo import GraphAlgo


class TestGraphAlgo(TestCase):

    def grpahs_equal(self, graph1: DiGraph, graph2: DiGraph):
        if graph1.v_size() != graph2.v_size() or graph1.e_size() != graph2.e_size():
            return False
        nodes_1 = graph1.get_all_v()
        nodes_2 = graph2.get_all_v()
        for node_1 in nodes_1:
            if node_1 not in nodes_2:
                return False
            else:
                node_2 = nodes_2[node_1].get_id()
                node_1_edges = graph1.all_out_edges_of_node(node_1).items()
                node_2_edges = graph2.all_out_edges_of_node(node_2).items()
                for edge_1 in node_1_edges:
                    if edge_1 not in node_2_edges:
                        return False
        return True

    def test_get_graph(self):
        graph_d = DiGraph()
        graph_d.add_node(0)
        graph_d.add_node(1)
        graph_d.add_node(2)
        graph_d.add_node(3)
        graph_d.add_node(4)
        graph_d.add_edge(0, 1, 6)
        graph_d.add_edge(0, 2, 9)
        graph_d.add_edge(1, 2, 2)
        graph_d.add_edge(1, 3, 7)
        graph_d.add_edge(1, 4, 5)
        graph_d.add_edge(2, 0, 3)
        graph_d.add_edge(2, 3, 1)
        graph_d.add_edge(3, 4, 1)
        graph_d.add_edge(4, 1, 3)
        graph_a = GraphAlgo(graph_d)
        graph_b = copy.deepcopy(graph_d)
        self.assertTrue(self.grpahs_equal(graph_b, graph_a.get_graph()))
        graph_b.remove_node(0)
        graph_b.add_node(6)
        self.assertFalse(self.grpahs_equal(graph_b, graph_a.get_graph()))

    def test_save_to_json(self):
        graph_d = DiGraph()
        graph_d.add_node(0)
        graph_d.add_node(1)
        graph_d.add_node(2)
        graph_d.add_node(3)
        graph_d.add_node(4)
        graph_d.add_edge(0, 1, 6)
        graph_d.add_edge(0, 2, 9)
        graph_d.add_edge(1, 2, 2)
        graph_d.add_edge(1, 3, 7)
        graph_d.add_edge(1, 4, 5)
        graph_d.add_edge(2, 0, 3)
        graph_d.add_edge(2, 3, 1)
        graph_d.add_edge(3, 4, 1)
        graph_d.add_edge(4, 1, 3)
        graph_a = GraphAlgo(graph_d)
        self.assertTrue(graph_a.save_to_json("test1.txt"))

    def test_load_from_json(self):
        graph_d = DiGraph()
        graph_d.add_node(0)
        graph_d.add_node(1)
        graph_d.add_node(2)
        graph_d.add_node(3)
        graph_d.add_node(4)
        graph_d.add_edge(0, 1, 6)
        graph_d.add_edge(0, 2, 9)
        graph_d.add_edge(1, 2, 2)
        graph_d.add_edge(1, 3, 7)
        graph_d.add_edge(1, 4, 5)
        graph_d.add_edge(2, 0, 3)
        graph_d.add_edge(2, 3, 1)
        graph_d.add_edge(3, 4, 1)
        graph_d.add_edge(4, 1, 3)
        graph_a = GraphAlgo(graph_d)
        self.assertTrue(graph_a.save_to_json("test1.txt"))
        self.assertTrue(graph_a.load_from_json("test1.txt"))
        self.assertFalse(graph_a.load_from_json("test2.txt"))  # Attempt to load the graph from a non-existent file.

    def test_shortest_path(self):
        graph_d = DiGraph()
        graph_d.add_node(0)
        graph_d.add_node(1)
        graph_d.add_node(2)
        graph_d.add_node(3)
        graph_d.add_node(4)
        graph_d.add_edge(0, 1, 6)
        graph_d.add_edge(0, 2, 9)
        graph_d.add_edge(1, 2, 2)
        graph_d.add_edge(1, 3, 7)
        graph_d.add_edge(1, 4, 5)
        graph_d.add_edge(2, 0, 3)
        graph_d.add_edge(2, 3, 1)
        graph_d.add_edge(3, 4, 1)
        graph_d.add_edge(4, 1, 3)
        graph_a = GraphAlgo(graph_d)
        self.assertTupleEqual((5, [1, 2, 0]), graph_a.shortest_path(1, 0))  # Finds a path between 2 connected nodes.
        self.assertTupleEqual((10, [0, 1, 2, 3, 4]),
                              graph_a.shortest_path(0, 4))  # Finds a path between 2 connected nodes.
        self.assertTupleEqual((3, [1, 2, 3]), graph_a.shortest_path(1, 3))  # Finds a path between 2 connected nodes.
        self.assertTrue(graph_a.get_graph().remove_edge(2, 0))  # Removes an edge.
        self.assertTupleEqual((float('inf'), []), graph_a.shortest_path(3,
                                                                        0))  # Attempt to find a path between 2 nodes which exist in the graph but not connected.
        self.assertTupleEqual((float('inf'), []), graph_a.shortest_path(4,
                                                                        7))  # Attempt to find a path between a node which exists in the graph and one which not exists in the graph.
        self.assertTupleEqual((float('inf'), []), graph_a.shortest_path(8,
                                                                        1))  # Attempt to find a path between a node which not exists in the graph and one which exists in the graph.
        self.assertTupleEqual((float('inf'), []), graph_a.shortest_path(13,
                                                                        11))  # Attempt to find a path between 2 nodes which not exist in the graph.

        self.assertTupleEqual((0, [3]), graph_a.shortest_path(3, 3))

    def test_connected_component(self):
        graph_d = DiGraph()
        graph_a = GraphAlgo(graph_d)
        self.assertListEqual([], graph_a.connected_component(1))
        graph_a.get_graph().add_node(1)
        graph_a.get_graph().add_node(2)
        graph_a.get_graph().add_node(3)
        graph_a.get_graph().add_node(4)
        graph_a.get_graph().add_node(5)
        graph_a.get_graph().add_node(6)
        graph_a.get_graph().add_node(7)
        graph_a.get_graph().add_node(8)
        graph_a.get_graph().add_edge(1, 2, 6)
        graph_a.get_graph().add_edge(3, 1, 9)
        graph_a.get_graph().add_edge(2, 3, 2)
        graph_a.get_graph().add_edge(2, 4, 7)
        graph_a.get_graph().add_edge(4, 1, 5)
        graph_a.get_graph().add_edge(5, 8, 3)
        graph_a.get_graph().add_edge(7, 6, 1)
        graph_a.get_graph().add_edge(6, 5, 1)
        graph_a.get_graph().add_edge(8, 6, 3)
        self.assertListEqual([], graph_a.connected_component(
            10))  # Attempt to find the SCC that a node which not exists in the graph should be a part of it.
        self.assertListEqual([1, 2, 3, 4], graph_a.connected_component(1))
        self.assertListEqual([5, 8, 6], graph_a.connected_component(5))
        self.assertListEqual([7], graph_a.connected_component(7))
        graph_a.get_graph().add_node(9)
        self.assertListEqual([9], graph_a.connected_component(9))
        self.assertListEqual([3, 1, 2, 4], graph_a.connected_component(3))

    def test_connected_components(self):
        graph_d = DiGraph()
        graph_a = GraphAlgo(graph_d)
        graph_aa = GraphAlgo()
        self.assertListEqual([], graph_a.connected_component(1))
        graph_a.get_graph().add_node(1)
        graph_a.get_graph().add_node(2)
        graph_a.get_graph().add_node(3)
        graph_a.get_graph().add_node(4)
        graph_a.get_graph().add_node(5)
        graph_a.get_graph().add_node(6)
        graph_a.get_graph().add_node(7)
        graph_a.get_graph().add_node(8)
        graph_a.get_graph().add_edge(1, 2, 6)
        graph_a.get_graph().add_edge(3, 1, 9)
        graph_a.get_graph().add_edge(2, 3, 2)
        graph_a.get_graph().add_edge(2, 4, 7)
        graph_a.get_graph().add_edge(4, 1, 5)
        graph_a.get_graph().add_edge(5, 8, 3)
        graph_a.get_graph().add_edge(7, 6, 1)
        graph_a.get_graph().add_edge(6, 5, 1)
        graph_a.get_graph().add_edge(8, 6, 3)
        counter = 0
        for component in graph_a.connected_components():
            if counter == 0:
                self.assertTrue(1 in component and 2 in component and 3 in component and 4 in component)
            if counter == 2:
                self.assertTrue(7 in component)
            if counter == 1:
                self.assertTrue(5 in component and 6 in component and 8 in component)
            counter += 1
        print(graph_a.connected_components())

    def test_plot_graph(self):
        graph_d = DiGraph()
        graph_d.add_node(1)
        graph_d.add_node(2)
        graph_d.add_node(3)
        graph_d.add_node(4)
        graph_d.add_node(5)
        graph_d.add_edge(1, 2, 3)
        graph_d.add_edge(2, 1, 4)
        graph_d.add_edge(1, 3, 4)
        graph_d.add_edge(3, 1, 6)
        graph_d.add_edge(3, 5, 1.2)
        graph_d.add_edge(2, 4, 6)
        graph_a = GraphAlgo(graph_d)
        self.assertIsNone(graph_a.plot_graph())


if __name__ == '__main__':
    unittest.main()
