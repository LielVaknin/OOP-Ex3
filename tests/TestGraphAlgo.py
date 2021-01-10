import time
import unittest
from unittest import TestCase

from DiGraph import DiGraph
from GraphAlgo import GraphAlgo


class TestGraphAlgo(TestCase):

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
        self.assertEqual(graph_d, graph_a.get_graph())

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
        self.assertTupleEqual((float('inf'), []), graph_a.shortest_path(3,
                                                                        3))  # Attempt to find a path between a node in the graph to itself.
        self.assertTupleEqual((float('inf'), []), graph_a.shortest_path(3,
                                                                        3))  # Attempt to find a path between a node which not exists in the graph to itself.

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
        self.assertListEqual([5, 6, 8], graph_a.connected_component(5))
        self.assertListEqual([7], graph_a.connected_component(7))
        graph_a.get_graph().add_node(9)
        self.assertListEqual([9], graph_a.connected_component(9))
        self.assertListEqual([1, 2, 3, 4], graph_a.connected_component(3))

    def test_connected_components(self):
        graph_d = DiGraph()
        graph_a = GraphAlgo(graph_d)
        graph_aa=GraphAlgo()
        graph_aa.load_from_json("10^4 nodes with 10^5 edges.json")
        start = time.time()
        tmp = graph_aa.connected_components()
        end = time.time()
        print("time: ", end - start)
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
        # self.assertListEqual([[1, 2, 3, 4], [5, 6, 8], [7]], graph_a.connected_components())
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
        self.assertIsNone(graph_a.plot_graph(graph_a))


if __name__ == '__main__':
    unittest.main()
