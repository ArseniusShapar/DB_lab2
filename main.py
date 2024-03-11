import sys

import numpy as np

Record = tuple[str, str]


def find_index(key: int, keys: list[int]) -> int:
    if not keys:
        return 0

    flag = False
    for i, k in enumerate(keys):
        if key < k:
            flag = True
            break
    if not flag:
        i = len(keys)

    return i


def hash_name(name: str) -> int:
    name = name.lower()
    step = 27
    chars_codes = np.array([int.from_bytes(str.encode(c)) - 97 for c in name])
    multipliers = np.array([float(sys.maxsize)] + [0.0 for _ in range(len(name) - 1)])

    for i in range(1, len(name)):
        multipliers[i] = multipliers[i - 1] / step

    h = np.sum(chars_codes * multipliers)
    return int(h)


names = ['Abigail', 'Ben', 'Courtney', 'David', 'Eline', 'Frank', 'Greg', 'Harper', 'Ivy', 'John', 'Kevin', 'Logan',
         'Mary', 'Noah', 'Olivia', 'Patrick', 'Quentin', 'Robert', 'Sara', 'Timmy', 'Ursule', 'Vincent', 'Wendy',
         'Xavier', 'Yonatan', 'Zoe']
phones = [f'+3800000000{"0" + str(i) if i < 10 else i}' for i in range(1, 27)]


class Node:
    def __init__(self, keys: list[int] = None, childs: list['Node', 'Leaf'] = None):
        if keys is None:
            keys = []
        if childs is None:
            childs = []

        self.min_order = 2
        self.max_order = 4
        self.keys = keys
        self.childs = childs

    def __str__(self) -> str:
        return str(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def is_owerflowed(self) -> bool:
        return len(self) > self.max_order

    def add_child(self, node: 'Node') -> None:
        key = node.keys[0]
        i = find_index(key, self.keys)
        self.keys.insert(i, key)
        self.childs.insert(i + 1, node)
        return

    def split(self) -> 'Node':
        mid = len(self.keys) // 2
        new_node = Node(self.keys[mid:], self.childs[mid + 1:])
        self.keys = self.keys[:mid]
        self.childs = self.childs[:mid + 1]
        return new_node

    def delete_child(self, child: 'Node') -> None:
        i = self.childs.index(child)
        self.keys.pop(i)
        self.childs.pop(i)


class Leaf(Node):
    def __init__(self, keys: list[int] = None,
                 values: list[Record] = None, neighbour: 'Leaf' = None):
        super().__init__(keys)
        # del self.childs

        if values is None:
            values = []

        self.neighbour = neighbour
        self.values = values

    def __str__(self) -> str:
        return str(self.keys) + ' ' + str(self.values)

    def insert(self, key: int, value: Record) -> None:
        i = find_index(key, self.keys)
        self.keys.insert(i, key)
        self.values.insert(i, value)

    def split(self) -> 'Leaf':
        mid = self.max_order // 2
        new_leaf = Leaf(self.keys[mid:], self.values[mid:], self.neighbour)
        self.keys = self.keys[:mid]
        self.values = self.values[:mid]
        self.neighbour = new_leaf
        return new_leaf

    def delete(self, key: int) -> None:
        i = self.keys.index(key)
        self.keys.pop(i)
        self.values.pop(i)


class Root(Node):
    pass


class BPlusTree:
    def __init__(self):
        self.root = Leaf()

    def _tab(self, n: int) -> str:
        return '\n' + n * '\t'

    def _show(self, node: Node, n: int = 0) -> str:
        return self._tab(n) + str(node) + ''.join([self._show(child, n + 1) for child in node.childs])

    def __str__(self) -> str:
        return self._show(self.root)

    def __search_leaf(self, key: int, node: Node) -> Leaf:
        if type(node) == Leaf:
            return node

        for i, k in enumerate(node.keys):
            if key < k:
                return self.__search_leaf(key, node.childs[i])

        return self.__search_leaf(key, node.childs[-1])

    def _search_leaf(self, key: int) -> Leaf:
        return self.__search_leaf(key, self.root)

    def search(self, name: str = '', key: int = None) -> list[str]:
        key = hash_name(name) if key is None else key
        leaf = self._search_leaf(key)
        values = [value for k, value in zip(leaf.keys, leaf.values) if k == key]
        leaf = leaf.neighbour
        end = False
        while (not end) and (leaf is not None):
            for k, value in zip(leaf.keys, leaf.values):
                if k == key:
                    values.append(value)
                else:
                    end = True
                    break
            leaf = leaf.neighbour

        return values

    def __left_leaf(self, node: Node) -> Leaf:
        if type(node) == Leaf:
            return node

        return self.__left_leaf(node.childs[0])

    def _left_leaf(self) -> Leaf:
        return self.__left_leaf(self.root)

    def search_left(self, name: str = '', key: int = None) -> list[str]:
        key = hash_name(name) if key is None else key
        leaf = self._left_leaf()
        values = []
        end = False
        while (not end) and (leaf is not None):
            for k, value in zip(leaf.keys, leaf.values):
                if k < key:
                    values.append(value)
                else:
                    end = True
                    break
            leaf = leaf.neighbour

        return values

    def search_right(self, name: str = '', key: int = None) -> list[str]:
        key = hash_name(name) if key is None else key
        leaf = self._search_leaf(key)
        values = [value for k, value in zip(leaf.keys, leaf.values) if k > key]
        leaf = leaf.neighbour

        while leaf is not None:
            values += [value for value in leaf.values]
            leaf = leaf.neighbour
        return values

    def _parent(self, node: Node) -> Node:
        current_node = self.root

        while True:
            for child in current_node.childs:
                if child is node:
                    return current_node

            flag = False
            for i, k in enumerate(current_node.keys):
                if node.keys[0] < k:
                    current_node = current_node.childs[i]
                    flag = True
                    break
            if not flag:
                current_node = current_node.childs[-1]

    def insert(self, name: str = '', phone: str = '', key: int = None) -> None:
        key = hash_name(name) if key is None else key
        leaf = self._search_leaf(key)

        leaf.insert(key, (name, phone))

        if not leaf.is_owerflowed():
            return

        new_leaf = leaf.split()

        if type(self.root) == Leaf:
            parent = Root()
            self.root = parent
            parent.childs = [leaf]
        else:
            parent = self._parent(leaf)

        parent.add_child(new_leaf)

        if not parent.is_owerflowed():
            return

        new_node = parent.split()

        if type(parent) != Root:
            self.root.add_child(new_node)
            new_node.keys.pop(0)
            return

        mid_key = new_node.keys.pop(0)
        new_root = Root([mid_key], [Node(self.root.keys, self.root.childs), new_node])
        self.root = new_root

    def delete(self, name: str = '', key: int = None) -> None:
        key = hash_name(name) if key is None else key
        leaf = self._search_leaf(key)

        leaf.delete(key)

        if (len(leaf) >= leaf.min_order) or (len(leaf) == 0):
            return

        will_insert = [(leaf.keys[0], leaf.values[0])]

        parent = self._parent(leaf)
        parent.delete_child(leaf)

        if (len(parent) < parent.min_order) and (type(parent) != Root):
            for leaf in parent.childs:
                will_insert += [(k, v) for k, v in zip(leaf.keys, leaf.values)]

            self.root.delete_child(parent)

        for k, v in will_insert:
            self.insert(name=v[0], phone=v[1], key=k)


if __name__ == '__main__':
    tree = BPlusTree()

    # for i, phone in enumerate(phones):
    #     tree.insert(key=i + 1, phone=phone)

    # for name, phone in zip(names, phones):
    #     tree.insert(name=name, phone=phone)

    for k in [22, 89, 16, 9, 88, 100, 40, 89, 84, 47, 56, 3, 88, 87, 96]:
        tree.insert(key=k, phone=phones[k % 26])


    print(tree)
    print()

    # tree.delete(key=0)

    # tree.delete(key=22)
    # tree.insert(key=56)
    print(tree.search_right(key=56))
