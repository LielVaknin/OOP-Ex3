import copy
import math
import random
from collections import deque
from queue import Queue
from matplotlib.patches import ConnectionPatch
from src.DiGraph import DiGraph
from src.GraphAlgoInterface import GraphAlgoInterface
import json
import heapq
import matplotlib.pyplot as plt


class GraphAlgo(GraphAlgoInterface):
    """This class represents a Directed Weighted Graph Theory algorithms.
       This object works on directed weighted graph and run some algorithms over it"""

    def __init__(self, graph: DiGraph = None):
        if graph is None:
            graph = DiGraph()
        self.__graph = graph

    def get_graph(self) -> DiGraph:
        """
         This method returns the graph which the Graph_Algo works on.
        :return: directed weighted graph.
        """
        return self.__graph

    def load_from_json(self, file_name: str) -> bool:
        """
        Loads a graph from a json file and inits it.
        This method returns True if the file successfully loaded, False o.w.
        @param file_name: Represents the path to the json file.
        @returns True if the loading was successful, False otherwise.
        """
        my_dict = {}
        graph = DiGraph()
        try:
            with open(file_name, "r")as file:
                my_dict = json.load(file)
                nodes = my_dict["Nodes"]
                edges = my_dict["Edges"]
                for node_dict in nodes:
                    if len(node_dict) < 2:
                        graph.add_node(node_dict["id"])
                    else:
                        graph.add_node(node_dict["id"], node_dict["pos"])
                for edge_dict in edges:
                    graph.add_edge(edge_dict["src"], edge_dict["dest"], edge_dict["w"])

            self.__graph = graph
            return True
        except IOError as e:
            print(e)
            return False

    def save_to_json(self, file_name: str):
        """
        Saves the graph at JSON format into a file(by given path name).
        @param file_name: Represents the path to the output file.
        @return: True if the save was successful, False otherwise.
        """
        nodes = []
        edges = []
        for node in self.get_graph().get_all_v().items():
            nodes_dict = {}
            nodes_dict["id"] = node[1].get_id()
            if len(node) > 1:
                nodes_dict["pos"] = node[1].get_pos()
            nodes.append(nodes_dict)
            node_edges = self.get_graph().all_out_edges_of_node(node[1].get_id())
            # node_edges_keys = self.get_graph().all_out_edges_of_node(node).keys()
            for edge in node_edges:
                edges_dict = {}
                edges_dict["src"] = node[1].get_id()
                edges_dict["dest"] = edge
                edges_dict["w"] = node_edges[edge]
                edges.append(edges_dict)
        ans_dict = {}
        ans_dict["Nodes"] = nodes
        ans_dict["Edges"] = edges

        try:
            with open(file_name, "w") as file:
                json.dump(ans_dict, default=lambda m: m.__dict__, indent=4, fp=file)
                return True
        except IOError as e:
            print(e)
            return False

    def shortest_path(self, id1: int, id2: int) -> (float, list):
        """
        Returns the shortest path from node id1 to node id2 using Dijkstra's Algorithm.
        @param id1: Represents the src node id.
        @param id2: Represents the dest node id.
        @return: The distance of the path and list of the nodes ids that the path goes through.
        If there is no path between id1 and id2, or one of them dose not exist the function returns (float('inf'),[]).
        """
        ans = (math.inf, [])
        graph = self.get_graph()
        graph_nodes = graph.get_all_v()
        visited = {}
        distance = {}
        path = {}
        queue = []
        if id1 not in graph_nodes or id2 not in graph_nodes:
            return ans
        if id1 == id2:
            ans = (0, [id1])
            return ans
        for node in graph.get_all_v():
            distance[node] = math.inf
        distance[id1] = 0
        heapq.heappush(queue, (distance[id1], id1))
        while len(queue) != 0:
            current = heapq.heappop(queue)[1]
            if current in visited:
                continue
            else:
                visited[current] = 1
                curr_edges = graph.all_out_edges_of_node(current)
                for tmp_node in curr_edges:
                    edge = curr_edges[tmp_node]
                    if edge + distance[current] < distance[tmp_node]:
                        distance[tmp_node] = edge + distance[current]
                        path[tmp_node] = current
                    if tmp_node not in visited:
                        heapq.heappush(queue, (distance[tmp_node], tmp_node))

        if id2 not in path:
            return ans

        final_distance = 0
        final_list = []
        final_list.append(id2)
        final_distance = distance[id2]
        tmp = path[id2]
        while tmp != id1:
            final_list.append(tmp)
            tmp = path[tmp]
        final_list.append(id1)
        final_list.reverse()
        return (final_distance, final_list)

    def connected_component(self, id1: int) -> list:
        """
        Finds the Strongly Connected Component(SCC) that node id1 is part of.
        @param id1: Represents the node id.
        @return: The list of nodes in the SCC.
        If the graph is None or id1 is not in the graph, the function should return an empty list [].
        """
        if self.get_graph() is None:
            return []
        graph = self.__graph
        ans = {}
        if id1 not in graph.get_all_v():
            return []
        ans = self.__dfs(id1)
        ans_2 = {}
        dic = graph.end_edges
        graph.end_edges = graph.edges
        graph.edges = dic
        ans_2 = self.__dfs(id1)
        intersection = []
        for node in ans:
            if node in ans_2:
                intersection.append(node)
        dic = graph.end_edges
        graph.end_edges = graph.edges
        graph.edges = dic
        # intersection.sort()
        return intersection

    def connected_components(self):
        """
        Finds all the Strongly Connected Component(SCC) in the graph.
        @return: The list of all SCC.
        If the graph is None the function should return an empty list [].
        """
        ans = []
        vis = {}
        if self.get_graph() is None:
            return ans
        for node in self.get_graph().get_all_v():
            if node not in vis:
                tmp_list = self.connected_component(node)
                for n in tmp_list:
                    vis[n] = 1
                ans.append(tmp_list)
        return ans

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner (by using generate_locations method).
        @return: None.
        """
        XV = []
        YV = []
        graph = self.get_graph()
        sum = 10 * graph.v_size()
        nodes = graph.get_all_v().items()

        max_x = 0
        max_y = 0
        min_x = math.inf
        min_y = math.inf
        text = []
        for node in nodes:
            if node[1].get_pos() is None:
                self.__generate_locations()
            node_x = float(node[1].get_pos().split(',')[0])
            node_y = float(node[1].get_pos().split(',')[1])
            if node_x > max_x:
                max_x = node_x
            if node_y > max_y:
                max_y = node_y
            if node_x < min_x:
                min_x = node_x
            if node_y < min_y:
                min_y = node_y
        frame_x = max_x - min_x
        frame_y = max_y - min_y
        rad = 1 / 100 * frame_y
        for node in nodes:
            node_x = float(node[1].get_pos().split(',')[0])
            node_y = float(node[1].get_pos().split(',')[1])
            XV.append(node_x)
            YV.append(node_y)
            text.append([node_x + rad, node_y + rad, node[1].get_id()])
            for edge in graph.all_out_edges_of_node(node[0]):
                dest = graph.get_all_v()[edge]
                dest_x = float(dest.get_pos().split(',')[0])
                dest_y = float(dest.get_pos().split(',')[1])
                dx = dest_x - node_x
                dy = dest_y - node_y
                line_w = 0.0002 * frame_x
                if line_w > 0.2 * frame_y:
                    line_w = 0.2 * frame_y

                plt.arrow(node_x, node_y, dx, dy, width=line_w, length_includes_head=True, head_width=30 * line_w,
                          head_length=75 * line_w, color='k')

        # plt.text()
        for tex in text:
            plt.text(tex[0], tex[1], tex[2], color='b')

        plt.plot(XV, YV, 'o', color='r')
        plt.grid()
        plt.title("Graph")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def __generate_locations(self):
        sum = self.get_graph().v_size() + 10
        counter = 1
        graph = self.get_graph()
        for node in graph.get_all_v():
            x = counter / sum
            y = random.random()
            z = 0
            pos = str(x) + ',' + str(y) + ',' + str(z)
            graph.get_all_v()[node].set_pos(pos)
            counter += 1

    def __dfs(self, id1: int) -> dict:
        graph = self.__graph
        ans = {id1: 1}
        vis = {}
        stack = deque()
        stack.append(id1)
        while len(stack) != 0:
            curr = stack.pop()
            if curr not in vis:
                vis[curr] = 1
                for neighbor in graph.all_out_edges_of_node(curr):
                    if neighbor not in vis:
                        stack.append(neighbor)
                        ans[neighbor] = 1
        return ans
