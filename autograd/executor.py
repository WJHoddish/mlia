from node import Node


class Executor:

    def __init__(self, root):
        self.topo = self.__topological_sorting(root)  # 拓扑排序的顺序就是正向求值的顺序
        self.root = root

    def run(self):
        """
        按照拓扑排序的顺序对计算图求值。注意：因为我们之前对node采用了eager模式，
        实际上每个node值之前已经计算好了，但为了演示lazy计算的效果，这里使用拓扑
        排序又计算了一遍。
        """
        node_evaluated = set()  # 保证每个node只被求值一次

    def __dfs(self, topo, node):
        if node is None or not isinstance(node, Node):
            return

        for n in node.x:
            self.__dfs(topo, n)  # recursion

        topo.append(node)

    def __topological_sorting(self, root):
        topo = []

        # start recursion
        self.__dfs(topo, root)

        return topo
