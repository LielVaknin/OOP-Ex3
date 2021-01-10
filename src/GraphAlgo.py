import copy
import math
from collections import deque
from queue import Queue
from matplotlib.patches import ConnectionPatch
from DiGraph import DiGraph
from GraphAlgoInterface import GraphAlgoInterface
import json
import heapq
import matplotlib.pyplot as plt


class GraphAlgo(GraphAlgoInterface):
    """This class represents a Directed Weighted Graph Theory algorithms."""

    def __init__(self, graph: DiGraph = None):
        if graph is None:
            graph = DiGraph()
        self.__graph = graph

    def get_graph(self) -> DiGraph:
        """
        :return: directed weighted graph on which the algorithm works on.
        """
        return self.__graph

    def load_from_json(self, file_name: str) -> bool:
        """
        Loads a graph from a json file.
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
        Saves the graph in JSON format to a file.
        @param file_name: Represents the path to the out file.
        @return: True if the save was successful, False otherwise.
        """
        nodes = []
        edges = []
        for node in self.get_graph().get_all_v():
            nodes_dict = {}
            nodes_dict["id"] = node
            nodes.append(nodes_dict)
            node_edges = self.get_graph().all_out_edges_of_node(node)
            node_edges_keys = self.get_graph().all_out_edges_of_node(node).keys()
            for edge in node_edges_keys:
                edges_dict = {}
                edges_dict["src"] = node
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
        @param id1: Represents the start node id.
        @param id2: Represents the end node id.
        @return: The distance of the path, a list of the nodes ids that the path goes through.
        If there is no path between id1 and id2, or one of them dose not exist the function returns (float('inf'),[]).
        """
        ans = (math.inf, [])
        graph = self.get_graph()
        graph_nodes = graph.get_all_v()
        visited = {}
        distance = {-1: math.inf}
        path = {}
        queue = []
        if id1 not in graph_nodes or id2 not in graph_nodes or id1 == id2:
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
        Finds the Strongly Connected Component(SCC) that node id1 is a part of.
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
        @return: The list all SCC.
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
        Otherwise, they will be placed in a random but elegant manner.
        @return: None.
        """
        ax1 = plt.subplots()
        XV = []
        YV = []
        arrows_s_x = []
        arrows_s_y = []
        arrows_d_dx = []
        arrows_d_dy = []
        graph = self.get_graph()
        nodes = graph.get_all_v().items()
        for node in nodes:
            node_x = float(node[1].get_pos().split(',')[0])
            node_y = float(node[1].get_pos().split(',')[1])
            XV.append(node_x)
            YV.append(node_y)
            for edge in graph.all_out_edges_of_node(node[0]):
                dest = graph.get_all_v()[edge]
                dest_x = float(dest.get_pos().split(',')[0])
                dest_y = float(dest.get_pos().split(',')[1])
                dx = dest_x - node_x
                dy = dest_y - node_y
                plt.arrow(node_x, node_y, dx, dy, linewidth=0.2)

                # plt.arrow(arrows_s_x, arrows_s_y, arrows_d_dx, arrows_d_dy)

                # arrows_s_x.append(node_x)
                # arrows_s_y.append(node_y)
                # arrows_d_dx.append(dx)
                # arrows_d_dy.append(dy)

        # plt.Arrow()
        plt.plot(XV, YV, 'o')
        plt.grid()
        plt.title("Graph")
        plt.xlabel("x title")
        plt.ylabel("y title")
        plt.show()
        # xyA = (0.2, 0.2)
        # xyB = (0.8, 0.8)
        # coordsA = "data"
        # coordsB = "data"
        # con = ConnectionPatch(xyA, xyB, coordsA, coordsB,
        #                       arrowstyle="-|>", shrinkA=5, shrinkB=5,
        #                       mutation_scale=20, fc="w")
        # ax1.add_artist(con)
        # ax1.show()

    # def BFS(self, id1: int) -> list:
    #     graph = self.__graph
    #     ans = [id1]
    #     vis = {}
    #     vis[id1] = 1
    #     queue = Queue()
    #     queue.put(id1)
    #     while not queue.empty():
    #         curr = queue.get()
    #         vis[curr] = 1
    #         for neighbor in graph.all_out_edges_of_node(curr):
    #             if neighbor not in vis:
    #                 ans.append(neighbor)
    #                 queue.put(neighbor)
    #                 vis[neighbor] = 1
    #
    #     return ans

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

# nodes = graph.get_all_v()
# for node in nodes:
#     node_o = nodes[node]
#     node_x = float(node_o[1].get_pos().split(',')[0])
#     node_y = float(node_o[1].get_pos().split(',')[1])
#     XV.append(node_x)
#     YV.append(node_y)
#     for edge in graph.all_out_edges_of_node(node):
#         dest = nodes[edge]
#         dest_x = float(dest[1].get_pos().split(',')[0])
#         dest_y = float(dest[1].get_pos().split(',')[1])
#         dx = dest_x - node_x
#         dy = dest_y - node_y
#         plt.arrow(node_x, node_y, dx, dy)
