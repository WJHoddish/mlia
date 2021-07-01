class Node:
    """
    Node, where to store all values.
    """

    global_id = -1

    def __init__(self, edge, x):
        """

        :param edge: operation
        :param x: inputs ([a, b])
        """

        # operation that generate this node
        self.edge = edge
        self.grad = 0.0

        # inputs
        self.x = x

        self.value = None
        self.valuate()

        self.id = Node.global_id
        Node.global_id += 1

    def valuate(self):
        self.value = self.edge.compute(self.convert())

    def convert(self):
        x = []

        for i in self.x:
            if isinstance(i, Node):
                i = i.value
            x.append(i)

        return x
