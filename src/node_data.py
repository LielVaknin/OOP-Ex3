class NodeData:
    """
    This class represents the set of operations applicable on a
    node (vertex) in a directed weighted graph.
    """

    def __init__(self, key, pos):
        self.__key = key
        self.__tag = 0
        self.__pos = pos

    def get_id(self):
        """
         Returns the key (id) associated with this node.
        :return: key.
        """
        return self.__key

    def get_tag(self):
        """
        Returns temporal data which can be used by algorithms.
        :return: tag.
        """
        return self.__tag

    def get_pos(self):
        """
        Returns the location of this node.
        :return: pos.
        """
        return self.__pos

    def __str__(self):
        return f'key:{self.__key}'

    def __copy__(self):
        new_node = NodeData(self.__key, self.__pos)
        return new_node

    def set_tag(self, tag: int):
        self.__tag = tag
        return 0
