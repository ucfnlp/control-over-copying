class Node:
    def __init__(self, key, value):
        self.child = [None, None]
        self.parent = None
        self.size = 1
        self.key = key
        self.value = value


class Splay:
    def __init__(self, limit=10000):
        self.root = None
        self.limit = limit

    def breakLink(self, x):
        x.child = [None, None]
        x.parent = None

    def get_size(self, node):
        if node is None:
            return 0
        return node.size

    def size(self):
        return self.get_size(self.root)

    def rotate(self, x, d):
        y = x.child[1 - d]
        if y is not None:
            x.child[1 - d] = y.child[d]
            if (y.child[d] is not None):
                y.child[d].parent = x
            y.parent = x.parent

        if x.parent is None:
            self.root = y
        else:
            x.parent.child[d ^ int(x != x.parent.child[d])] = y
        if (y is not None):
            y.child[d] = x
        x.parent = y
        x.size = self.get_size(x.child[0]) + self.get_size(x.child[1]) + 1
        if (y is not None):
            y.size = self.get_size(y.child[0]) + self.get_size(y.child[1]) + 1

    def splay(self, x):
        while x.parent is not None:
            if x.parent.parent is None:
                self.rotate(x.parent, int(x == x.parent.child[0]))
            else:
                d1 = x == x.parent.child[0]
                d2 = x.parent == x.parent.parent.child[0]
                if (d1 ^ d2):
                    self.rotate(x.parent, int(d1))
                else:
                    self.rotate(x.parent.parent, int(d1))
                self.rotate(x.parent, int(d2))

    def optimum(self, node, d):
        while node.child[d] is not None:
            node = node.child[d]
        self.splay(node)
        return node

    def minimum(self):
        return self.optimum(self.root, 0)

    def maximum(self):
        return self.optimum(self.root, 1)

    def maintain_size(self):
        if self.get_size(self.root) > self.limit:
            x = self.maximum()
            self.root = x.child[0]
            if (self.root is not None):
                self.root.parent = None
            self.breakLink(x)

    def push(self, key, value):
        x = self.root
        y = None
        while x is not None:
            y = x
            x = x.child[int(key >= x.key)]
        node = Node(key, value)
        node.parent = y
        if y is None:
            self.root = node
        else:
            y.child[int(node.key >= y.key)] = node
        self.splay(node)
        self.maintain_size()

    def isNotEmpty(self):
        return self.get_size(self.root) > 0

    def pop(self):
        x = self.minimum()
        self.root = x.child[1]
        if self.root is not None:
            self.root.parent = None
        self.breakLink(x)
        return x
