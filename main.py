import sys
from copy import deepcopy
import numpy as np

Record = tuple[str, str]


def find_index(key: int, keys: list[int]) -> int:
    if not keys:
        return 0

    if key >= keys[-1]:
        return len(keys)

    if key < keys[0]:
        return 0

    for i in range(len(keys) - 1):
        if keys[i] <= key < keys[i + 1]:
            return i + 1


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
    def __init__(self, keys: list[int] = None, childs: list['Node'] = None):
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
        i = self.min_order if len(set(self.keys[len(self.keys) // 2 + 1:])) == 1 else 0
        mid = i if i >= self.min_order else len(self.keys) // 2
        new_node = Node(self.keys[mid:], self.childs[mid + 1:])
        self.keys = self.keys[:mid]
        self.childs = self.childs[:mid + 1]
        return new_node

    def delete_child(self, child: 'Node') -> None:
        i = min((self.childs.index(child), len(self.keys) - 1))
        if self.keys:
            self.keys.pop(i)
        deleted_child: Leaf = self.childs.pop(i)
        if type(deleted_child) == Leaf:
            if deleted_child.left_neighbour is not None:
                deleted_child.left_neighbour.right_neighbour = deleted_child.right_neighbour
            if deleted_child.right_neighbour is not None:
                deleted_child.right_neighbour.left_neighbour = deleted_child.left_neighbour
        del deleted_child


class Leaf(Node):
    def __init__(self, keys: list[int] = None,
                 values: list[Record] = None,
                 left_neighbour: 'Leaf' = None,
                 right_neighbour: 'Leaf' = None):
        super().__init__(keys)
        # del self.childs

        if values is None:
            values = []

        self.left_neighbour = left_neighbour
        self.right_neighbour = right_neighbour
        self.values = values

    def __str__(self) -> str:
        return str(self.keys) + ' ' + str(self.values)

    def insert(self, key: int, value: Record) -> None:
        i = find_index(key, self.keys)
        self.keys.insert(i, key)
        self.values.insert(i, value)

    def split(self) -> 'Leaf':
        i = self.min_order if len(set(self.keys[len(self.keys) // 2 + 1:])) == 1 else 0
        mid = i if i >= self.min_order else len(self.keys) // 2
        new_leaf = Leaf(self.keys[mid:], self.values[mid:], left_neighbour=self, right_neighbour=self.right_neighbour)
        if self.right_neighbour is not None:
            self.right_neighbour.left_neighbour = new_leaf
        self.keys = self.keys[:mid]
        self.values = self.values[:mid]
        self.right_neighbour = new_leaf
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

        i = find_index(key, node.keys)
        return self.__search_leaf(key, node.childs[i])

    def _search_leaf(self, key: int) -> Leaf:
        return self.__search_leaf(key, self.root)

    def _search(self, leaf: Leaf, key: int, direct=-1, condition=lambda x, y: x == y) -> list[Record]:
        values = []
        while leaf is not None:
            if all([not (condition(k, key) or key == k) for k in leaf.keys]):
                break
            values += [value for k, value in zip(leaf.keys, leaf.values) if condition(k, key)]
            leaf = leaf.left_neighbour if direct == -1 else leaf.right_neighbour if direct == 1 else None
        return values

    def search(self, name: str = '', key: int = None) -> list[Record]:
        key = hash_name(name) if key is None else key
        leaf = self._search_leaf(key)
        values = self._search(leaf, key, direct=-1)
        values += self._search(leaf.right_neighbour, key, direct=1)
        values.sort(key=lambda item: item[0])
        return values

    def search_left(self, name: str = '', key: int = None) -> list[Record]:
        key = hash_name(name) if key is None else key
        leaf = self._search_leaf(key)
        values = self._search(leaf, key, direct=-1, condition=lambda x, y: x < y)
        values.sort(key=lambda item: item[0])
        return values

    def search_right(self, name: str = '', key: int = None) -> list[Record]:
        key = hash_name(name) if key is None else key
        leaf = self._search_leaf(key)
        values = self._search(leaf, key, direct=1, condition=lambda x, y: x > y)
        values.sort(key=lambda item: item[0])
        return values

    def _parent(self, node: Node) -> Node:
        current_node = self.root

        while True:
            for child in current_node.childs:
                if child is node:
                    return current_node

            i = find_index(node.keys[0], current_node.keys)
            current_node = current_node.childs[i]

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

        if (type(parent) == Root) or (len(self.root) == self.root.max_order):
            mid_key = new_node.keys.pop(0)
            new_root = Root([mid_key], [Node(self.root.keys, self.root.childs), new_node])
            self.root = new_root
            return

        self.root.add_child(new_node)
        new_node.keys.pop(0)

    def delete(self, name: str = '', key: int = None) -> None:
        key = hash_name(name) if key is None else key
        leaf = self._search_leaf(key)

        leaf.delete(key)

        if len(leaf) >= leaf.min_order:
            return

        will_insert = [(leaf.keys[0], leaf.values[0])]

        parent = self._parent(leaf)
        parent.delete_child(leaf)

        if (len(parent) == 0) and (type(parent) == Root):
            self.root = self.root.childs[0]

        if (len(parent) < parent.min_order) and (type(parent) != Root):
            for leaf in parent.childs:
                will_insert += [(k, v) for k, v in zip(leaf.keys, leaf.values)]
            for leaf in deepcopy(parent.childs):
                parent.delete_child(leaf)
            self.root.delete_child(parent)

        for k, v in will_insert:
            self.insert(name=v[0], phone=v[1], key=k)


if __name__ == '__main__':
    tree = BPlusTree()

    # for i, phone in enumerate(phones):
    #     tree.insert(key=i + 1, phone=phone)

    for name, phone in zip(names, phones):
        tree.insert(name=name, phone=phone)

    print(tree)
    print()
