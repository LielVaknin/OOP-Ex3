from unittest import TestCase
import unittest

from DiGraph import DiGraph


class TestDiGraph(TestCase):

    def test_v_size(self):
        graph1 = DiGraph()
        self.assertEqual(0, graph1.v_size())
        self.assertEqual(0, graph1.get_mc())
        graph1.add_node(1)
        graph1.add_node(2)
        graph1.add_node(3)
        graph1.add_node(4)
        graph1.add_node(5)
        self.assertEqual(5, graph1.v_size())
        self.assertEqual(5, graph1.get_mc())
        graph1.remove_node(5)
        graph1.remove_node(5)  # Attempt to delete a node that doesn't exist in the graph anymore.
        self.assertEqual(4, graph1.v_size())
        self.assertEqual(6, graph1.get_mc())

    def test_e_size(self):
        graph2 = DiGraph()
        self.assertEqual(0, graph2.e_size())
        self.assertEqual(0, graph2.get_mc())
        graph2.add_node(1)
        graph2.add_node(2)
        graph2.add_node(3)
        graph2.add_node(4)
        graph2.add_node(5)
        graph2.add_edge(1, 2, 3)
        graph2.add_edge(1, 3, 4)
        graph2.add_edge(2, 1, 4.7)
        graph2.add_edge(3, 2, 5.2)
        graph2.add_edge(4, 5, 9.5)
        self.assertEqual(5, graph2.e_size())
        self.assertEqual(10, graph2.get_mc())
        graph2.remove_edge(1, 3)
        graph2.remove_edge(1, 3)  # Attempt to delete an edge that doesn't exist in the graph anymore.
        self.assertEqual(4, graph2.e_size())
        self.assertEqual(11, graph2.get_mc())

    def test_get_all_v(self):
        graph3 = DiGraph()
        graph3.add_node(1)
        graph3.add_node(2)
        graph3.add_node(3)
        graph3.add_node(4)
        graph3.add_node(5)
        graph3.add_edge(1, 2, 3)
        graph3.add_edge(1, 3, 4)
        graph3.add_edge(2, 1, 4.7)
        graph3.add_edge(3, 2, 5.2)
        graph3.add_edge(4, 5, 9.5)
        graph3.add_edge(1, 2, 3)  # Attempt to add an edge that already exists in the graph.
        graph3.remove_node(3)
        dict_nodes = graph3.get_all_v()
        counter = 0
        for n in dict_nodes.keys():
            self.assertIsNotNone(dict_nodes[n])
            counter += 1
        self.assertEqual(counter, graph3.v_size())
        self.assertTrue(1 in dict_nodes)
        self.assertTrue(2 in dict_nodes)
        self.assertFalse(3 in dict_nodes)
        self.assertTrue(4 in dict_nodes)
        self.assertTrue(5 in dict_nodes)

    def test_all_in_edges_of_node(self):
        graph4 = DiGraph()
        graph4.add_node(1)
        graph4.add_node(2)
        graph4.add_node(3)
        graph4.add_node(4)
        graph4.add_node(5)
        graph4.add_edge(1, 2, 3)
        graph4.add_edge(1, 3, 4)
        graph4.add_edge(2, 1, 4.7)
        graph4.add_edge(3, 2, 5.2)
        graph4.add_edge(4, 5, 9.5)
        graph4.add_edge(1, 2, 3)  # Attempt to add an edge that already exists in the graph.
        graph4.remove_node(3)
        dict_nodes1 = graph4.all_in_edges_of_node(2)
        counter = 0
        for n in dict_nodes1.keys():
            self.assertIsNotNone(dict_nodes1[n])
            counter += 1
        self.assertEqual(1, counter)
        self.assertTrue(1 in dict_nodes1)
        self.assertFalse(2 in dict_nodes1)
        self.assertFalse(3 in dict_nodes1)
        self.assertFalse(4 in dict_nodes1)
        self.assertFalse(5 in dict_nodes1)
        dict_nodes2 = graph4.all_in_edges_of_node(0)
        self.assertDictEqual({}, dict_nodes2)

    def test_all_out_edges_of_node(self):
        graph5 = DiGraph()
        graph5.add_node(1)
        graph5.add_node(2)
        graph5.add_node(3)
        graph5.add_node(4)
        graph5.add_node(5)
        graph5.add_edge(1, 2, 3)
        graph5.add_edge(1, 3, 4)
        graph5.add_edge(2, 1, 4.7)
        graph5.add_edge(3, 2, 5.2)
        graph5.add_edge(4, 5, 9.5)
        graph5.add_edge(1, 2, 3)  # Attempt to add an edge that already exists in the graph.
        graph5.remove_node(3)
        dict_nodes1 = graph5.all_in_edges_of_node(2)
        counter = 0
        for n in dict_nodes1.keys():
            self.assertIsNotNone(dict_nodes1[n])
            counter += 1
        self.assertEqual(1, counter)
        self.assertTrue(1 in dict_nodes1)
        self.assertFalse(2 in dict_nodes1)
        self.assertFalse(3 in dict_nodes1)
        self.assertFalse(4 in dict_nodes1)
        self.assertFalse(5 in dict_nodes1)
        dict_nodes2 = graph5.all_in_edges_of_node(0)
        self.assertDictEqual({}, dict_nodes2)

    def test_get_mc(self):
        graph6 = DiGraph()
        self.assertEqual(0, graph6.get_mc())
        graph6.add_node(1)
        graph6.add_node(2)
        self.assertEqual(2, graph6.get_mc())
        graph6.add_edge(1, 2, 3)
        self.assertEqual(3, graph6.get_mc())
        graph6.remove_edge(1, 2)
        self.assertEqual(4, graph6.get_mc())
        graph6.remove_node(1)
        self.assertEqual(5, graph6.get_mc())

    def test_add_edge(self):
        graph7 = DiGraph()
        self.assertEqual(0, graph7.e_size())
        self.assertEqual(0, graph7.get_mc())
        graph7.add_node(1)
        graph7.add_node(2)
        graph7.add_node(3)
        graph7.add_node(4)
        graph7.add_node(5)
        self.assertTrue(graph7.add_edge(1, 2, 3))
        self.assertTrue(graph7.add_edge(1, 3, 4))
        self.assertTrue(graph7.add_edge(2, 1, 4.7))
        self.assertTrue(graph7.add_edge(3, 2, 5.2))
        self.assertTrue(graph7.add_edge(4, 5, 9.5))
        self.assertEqual(5, graph7.e_size())
        self.assertEqual(10, graph7.get_mc())
        dict_nodes = graph7.all_out_edges_of_node(1)
        w1 = dict_nodes[2]
        self.assertEqual(3, w1)
        self.assertTrue(graph7.add_edge(4, 2, 8))  # Connects 2 nodes which there is no edge between them.
        self.assertEqual(6, graph7.e_size())
        self.assertEqual(11, graph7.get_mc())
        dict_nodes = graph7.all_out_edges_of_node(4)
        w2 = dict_nodes[2]
        self.assertEqual(8, w2)
        self.assertFalse(graph7.add_edge(6, 7, 6))  # Attempt to connect 2 nodes which don't exists in the graph.
        self.assertFalse(
            graph7.add_edge(4, 7, 5))  # Attempt to connect 2 nodes which only the first exists in the graph.
        self.assertFalse(
            graph7.add_edge(6, 4, 2.3))  # Attempt to connect 2 nodes which only the second exists in the graph.
        self.assertFalse(graph7.add_edge(2, 4, -5))  # Attempt to connect 2 nodes with weight < 0.
        self.assertFalse(graph7.add_edge(1, 1, 5))  # Attempt to connect a node to itself.
        self.assertEqual(6, graph7.e_size())  # The adding of edges failed as expected.
        self.assertEqual(11, graph7.get_mc())

    def test_add_node(self):
        graph8 = DiGraph()
        self.assertEqual(0, graph8.v_size())
        self.assertEqual(0, graph8.get_mc())
        self.assertTrue(graph8.add_node(1))
        self.assertTrue(graph8.add_node(2))
        self.assertEqual(2, graph8.v_size())
        self.assertEqual(2, graph8.get_mc())
        self.assertFalse(graph8.add_node(1))  # Attempt to add a node that already exists in the graph.
        self.assertEqual(2, graph8.v_size())
        self.assertEqual(2, graph8.get_mc())

    def test_remove_node(self):
        graph9 = DiGraph()
        self.assertEqual(0, graph9.v_size())
        self.assertEqual(0, graph9.e_size())
        self.assertEqual(0, graph9.get_mc())
        graph9.add_node(1)
        graph9.add_node(2)
        graph9.add_node(3)
        graph9.add_node(4)
        graph9.add_node(5)
        graph9.add_edge(1, 2, 3)
        graph9.add_edge(1, 3, 4)
        graph9.add_edge(2, 1, 4.7)
        graph9.add_edge(3, 2, 5.2)
        graph9.add_edge(4, 5, 9.5)
        self.assertEqual(5, graph9.v_size())
        self.assertEqual(5, graph9.e_size())
        self.assertEqual(10, graph9.get_mc())
        self.assertTrue(graph9.remove_node(1))
        self.assertEqual(4, graph9.v_size())
        self.assertEqual(2, graph9.e_size())
        self.assertEqual(14, graph9.get_mc())  # The increase of mc also caused because of edges deletion.
        self.assertFalse(graph9.remove_node(1))  # Attempt to remove node which no longer exists in the graph.
        self.assertEqual(4, graph9.v_size())
        self.assertEqual(2, graph9.e_size())
        self.assertEqual(14, graph9.get_mc())

    def test_remove_edge(self):
        graph10 = DiGraph()
        self.assertEqual(0, graph10.e_size())
        self.assertEqual(0, graph10.get_mc())
        graph10.add_node(1)
        graph10.add_node(2)
        graph10.add_node(3)
        graph10.add_node(4)
        graph10.add_node(5)
        graph10.add_edge(1, 2, 3)
        graph10.add_edge(1, 3, 4)
        graph10.add_edge(2, 1, 4.7)
        graph10.add_edge(3, 2, 5.2)
        graph10.add_edge(4, 5, 9.5)
        self.assertEqual(5, graph10.e_size())
        self.assertEqual(10, graph10.get_mc())
        self.assertTrue(graph10.remove_edge(1, 3))  # Removes the edge between 2 nodes which have an edge between them.
        self.assertEqual(4, graph10.e_size())
        self.assertEqual(11, graph10.get_mc())
        self.assertFalse(graph10.remove_edge(1, 4))  # Attempt to remove a non-existent edge.
        self.assertFalse(graph10.remove_edge(6, 7))  # Attempt to remove an edge between non-existent nodes.
        self.assertFalse(graph10.remove_edge(6,
                                             4))  # Attempt to remove an edge between 2 nodes which the first is not exists in the graph.
        self.assertFalse(graph10.remove_edge(4,
                                             7))  # Attempt to remove an edge between 2 nodes which the second is not exists in the graph.
        self.assertFalse(graph10.remove_edge(1, 1))  # Attempt to remove an edge from node to itself.

        self.assertEqual(4, graph10.e_size())  # The removals failed as expected.
        self.assertEqual(11, graph10.get_mc())


if __name__ == '__main__':
    unittest.main()
