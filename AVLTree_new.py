class Node:

    def __init__(self, key, parent, left, right, height, sum):
        self.key = key
        self.parent = parent
        self.left = left
        self.right = right
        self.height = height
        self.sum = sum

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if not self.right.height and not self.left.height:
            line = '%s' % self.key
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if not self.right.height:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.key
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if not self.left.height:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.key
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.key
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


class FakeNode:

    def __init__(self, add_attrs=False):
        self.height = 0
        self.sum = 0
        if add_attrs:
            self.left = FakeNode()
            self.right = FakeNode()


fake_node = FakeNode(add_attrs=True)


class AVLTree:

    def __init__(self, root=fake_node):
        if isinstance(root, Node):
            root.parent = None
        self.root = root

    def print_in_order(self):

        def in_order(r):
            if r.left.height:
                in_order(r.left)

            print('{:^12}{:^12}{:^12}{:^12}{:^12}{:^12}'.format(
                r.key,
                r.parent.key if r.parent else 'None',
                r.left.key if r.left.height else 'None',
                r.right.key if r.right.height else 'None',
                r.height, r.sum))

            if r.right.height:
                in_order(r.right)

        if self.root.height:
            in_order(self.root)

    def find(self, r, key):
        if not r.height:
            return 0
        if r.key == key:
            return 1
        elif r.key > key and r.left.height:
            return self.find(r.left, key)
        elif r.key < key and r.right.height:
            return self.find(r.right, key)
        return 0

    def update_height_and_sum(self, node, ancestors=False):
        node.height = 1 + max(node.left.height, node.right.height)
        node.sum = node.key + node.left.sum + node.right.sum

        if ancestors and node.parent:
            self.update_height_and_sum(node.parent, ancestors=True)

    def small_right_spin(self, node, parent, right):

        if right.left.height:
            node.right, right.left.parent = right.left, node
        else:
            node.right = fake_node

        right.left, node.parent = node, right

        if parent:
            if parent.left == node:
                parent.left, right.parent = right, parent
            else:
                parent.right, right.parent = right, parent
        else:
            self.root, right.parent = right, None

        self.update_height_and_sum(node, ancestors=True)

    def small_left_spin(self, node, parent, left):

        if left.right.height:
            node.left, left.right.parent = left.right, node
        else:
            node.left = fake_node

        left.right, node.parent = node, left

        if parent:
            if parent.right == node:
                parent.right, left.parent = left, parent
            else:
                parent.left, left.parent = left, parent
        else:
            self.root, left.parent = left, None

        self.update_height_and_sum(node, ancestors=True)

    def big_right_spin(self, node, parent, right, right_left):

        if right_left.left.height:
            node.right, right_left.left.parent = right_left.left, node
        else:
            node.right = fake_node

        if right_left.right.height:
            right.left, right_left.right.parent = right_left.right, right
        else:
            right.left = fake_node

        right_left.left, node.parent = node, right_left
        right_left.right, right.parent = right, right_left

        if parent:
            if parent.left == node:
                parent.left, right_left.parent = right_left, parent
            else:
                parent.right, right_left.parent = right_left, parent
        else:
            self.root, right_left.parent = right_left, None

        self.update_height_and_sum(right)
        self.update_height_and_sum(node, ancestors=True)

    def big_left_spin(self, node, parent, left, left_right):

        if left_right.right.height:
            node.left, left_right.right.parent = left_right.right, node
        else:
            node.left = fake_node

        if left_right.left.height:
            left.right, left_right.left.parent = left_right.left, left
        else:
            left.right = fake_node

        left_right.right, node.parent = node, left_right
        left_right.left, left.parent = left, left_right

        if parent:
            if parent.left == node:
                parent.left, left_right.parent = left_right, parent
            else:
                parent.right, left_right.parent = left_right, parent
        else:
            self.root, left_right.parent = left_right, None

        self.update_height_and_sum(left)
        self.update_height_and_sum(node, ancestors=True)

    def balance(self, node):

        if node.right.height - node.left.height > 1:
            if node.right.right.height >= node.right.left.height:
                self.small_right_spin(node, node.parent, node.right)

            else:
                self.big_right_spin(node, node.parent, node.right, node.right.left)

        elif node.left.height - node.right.height > 1:
            if node.left.left.height >= node.left.right.height:
                self.small_left_spin(node, node.parent, node.left)

            else:
                self.big_left_spin(node, node.parent, node.left, node.left.right)

        elif node.parent:
            return self.balance(node.parent)

    def find_to_insert(self, r, key):
        """Changes nothing if the key is in a set else inserts the key to the set."""

        if r.key > key:
            if r.left.height:
                self.find_to_insert(r.left, key)
            else:
                r.left = Node(key, r, fake_node, fake_node,
                              height=1, sum=key)
                self.update_height_and_sum(r, ancestors=True)
                self.balance(r)

        if r.key < key:
            if r.right.height:
                self.find_to_insert(r.right, key)
            else:
                r.right = Node(key, r, fake_node, fake_node,
                               height=1, sum=key)
                self.update_height_and_sum(r, ancestors=True)
                self.balance(r)

    def insert(self, key):
        if not self.root.height:
            self.root = Node(key, None, fake_node, fake_node,
                             height=1, sum=key)
        else:
            self.find_to_insert(self.root, key)

    def search_max(self, r):
        return self.search_max(r.right) if r.right.height else r

    def delete_leaf(self, r):
        if self.root == r:
            self.root = fake_node
            return

        if r.parent.left == r:
            r.parent.left = fake_node
        else:
            r.parent.right = fake_node

        self.update_height_and_sum(r.parent, ancestors=True)
        self.balance(r.parent)

    def delete_node_with_one_son(self, r):

        if r == self.root:
            if r.left.height:
                self.root, r.left.parent = r.left, None
            else:
                self.root, r.right.parent = r.right, None

            self.update_height_and_sum(self.root)
            self.balance(self.root)
            return

        if r.parent.left == r:
            if r.left.height:
                r.parent.left, r.left.parent = r.left, r.parent
            else:
                r.parent.left, r.right.parent = r.right, r.parent
        else:
            if r.left.height:
                r.parent.right, r.left.parent = r.left, r.parent
            else:
                r.parent.right, r.right.parent = r.right, r.parent

        self.update_height_and_sum(r.parent, ancestors=True)
        self.balance(r.parent)

    def delete_node_with_two_sons(self, r):
        max_node = self.search_max(r.left)  # max node from the left subtree
        r.key, max_node.key = max_node.key, r.key
        if max_node.left.height or max_node.right.height:
            self.delete_node_with_one_son(max_node)
        else:
            self.delete_leaf(max_node)

    def find_to_delete(self, r, key):
        """Changes nothing if the key isn't in a set else deletes the key from the set."""

        if r.key == key:
            if r.left.height and r.right.height:
                self.delete_node_with_two_sons(r)

            elif r.left.height or r.right.height:
                self.delete_node_with_one_son(r)

            else:
                self.delete_leaf(r)

            return

        if r.key > key and r.left.height:
            return self.find_to_delete(r.left, key)

        if r.key < key and r.right.height:
            return self.find_to_delete(r.right, key)

    def delete(self, key):
        if not self.root.height:
            return
        else:
            self.find_to_delete(self.root, key)


def merge_with_root(tree1, tree2, root):
    """Merges trees tree1, tree2 to one tree with root in root.  """

    tree = AVLTree(root)
    tree.root.left, tree.root.right = tree1.root, tree2.root

    if tree1.root:
        tree1.root.parent = tree.root

    if tree2.root:
        tree2.root.parent = tree.root

    return tree


def AVL_merge_with_root(tree1, tree2, root):
    """Merges AVL trees tree1, tree2 in one AVL tree with root in root.  """

    if abs(tree1.root.height - tree2.root.height) <= 1:

        tree = merge_with_root(tree1, tree2, root)
        tree.update_height_and_sum(tree.root)

        return tree

    elif tree1.root.height - tree2.root.height > 1:

        tree = AVL_merge_with_root(AVLTree(tree1.root.right), tree2, root)
        tree1.root.right, tree.root.parent = tree.root, tree1.root

        tree1.update_height_and_sum(tree1.root.right)
        tree1.update_height_and_sum(tree1.root)
        tree1.balance(tree1.root)

        return tree1

    elif tree1.root.height - tree2.root.height < 1:

        tree = AVL_merge_with_root(tree1, AVLTree(tree2.root.left), root)
        tree2.root.left, tree.root.parent = tree.root, tree2.root

        tree2.update_height_and_sum(tree2.root.left)
        tree2.update_height_and_sum(tree2.root)
        tree2.balance(tree2.root)

        return tree2


def merge(tree1, tree2):
    """
        Merges two trees tree1 and tree2 in one tree assuming that
        all elements of the tree1 are less than all elements of the tree2

    """
    if not tree1.root.height:
        return tree2

    if not tree2.root.height:
        return tree1

    new_root = tree1.search_max(tree1.root)
    tree1.delete(new_root.key)

    return AVL_merge_with_root(tree1, tree2, new_root)


def split(tree, k):
    """
        Splits the tree in two trees - tree1 containing elements less or equal k
        and tree2 containing elements greater than k.

    """
    if not tree.root.height:
        return AVLTree(), AVLTree()

    if tree.root.key > k:
        (tree1, tree2) = split(AVLTree(tree.root.left), k)
        tree2 = AVL_merge_with_root(tree2, AVLTree(tree.root.right), tree.root)

        return tree1, tree2

    else:
        (tree1, tree2) = split(AVLTree(tree.root.right), k)
        tree1 = AVL_merge_with_root(AVLTree(tree.root.left), tree1, tree.root)

        return tree1, tree2


def find_sum_on_segment(tree, left_point, right_point):
    """Returns sum of the tree elements contained in the segment [left_point; right_point].  """
    if not tree.root.height:
        return 0, tree

    (tree1, tree2) = split(tree, left_point - 1)
    (tree2, tree3) = split(tree2, right_point)

    if not tree2.root.height:
        res = 0
    else:
        res = tree2.root.sum

    tree = merge(tree1, merge(tree2, tree3))

    return res, tree


def test():
    import random
    import time

    n = 50000
    tree = AVLTree()
    acc = 0
    for i in range(n):
        val = random.randint(0, 10 ** 9)
        t0 = time.perf_counter()
        tree.insert(val)
        t1 = time.perf_counter()
        acc += (t1 - t0)
    for i in range(n):
        left_point = random.randint(0, 10 ** 9)
        right_point = random.randint(0, 10 ** 9)
        t0 = time.perf_counter()
        (res, tree) = find_sum_on_segment(tree, left_point, right_point)
        t1 = time.perf_counter()
        acc += (t1 - t0)

    print('Everything is OK. Execution time is {}'.format(acc))


def main():
    tree = AVLTree()
    s = 0
    f = lambda x: ((x % 1000000001) + (s % 1000000001)) % 1000000001
    n = int(input())

    for i in range(n):
        query, val = input().split(maxsplit=1)

        if query == '+':
            val = int(val)
            tree.insert(f(val))

        elif query == '-':
            val = int(val)
            tree.delete(f(val))

        elif query == '?':
            val = int(val)
            print('Found' if tree.find(tree.root, f(val)) else 'Not found')

        elif query == 's':
            left_point, right_point = map(int, val.split())
            (s, tree) = find_sum_on_segment(tree, f(left_point), f(right_point))
            print(s)

        #if tree.root.height:
            #tree.root.display()


def test2():
    tree = AVLTree()
    n = 20
    for i in range(n):
        tree = merge(tree, AVLTree(Node(i + 1, None, fake_node, fake_node, 1, i + 1)))
        tree.root.display()
        tree.print_in_order()
        print('-' * 72)



if __name__ == '__main__':
    #main()
    test()