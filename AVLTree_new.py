from display import Tree
import random
import time


class Node(Tree):

    def __init__(self, key, parent, left, right, height, sum):
        super().__init__(key, left, right, height)
        self.parent = parent
        self.sum = sum


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
        print('{:^12}{:^12}{:^12}{:^12}{:^12}{:^12}'.format(
            'key', 'parent', 'left', 'right', 'height', 'sum'))

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
    def check(*strings):
        tree = AVLTree()
        ans = tuple()
        s = 0

        f = lambda x: ((x % 1000000001) + (s % 1000000001)) % 1000000001

        for string in strings:
            query, val = string.split(maxsplit=1)

            if query == '+':
                val = int(val)
                tree.insert(f(val))

            elif query == '-':
                val = int(val)
                tree.delete(f(val))

            elif query == '?':
                val = int(val)
                ans += ('Found' if tree.find(tree.root, f(val)) else 'Not found', )

            elif query == 's':
                left_point, right_point = map(int, val.split())
                (s, tree) = find_sum_on_segment(tree, f(left_point), f(right_point))
                ans += (str(s), )

        return ans

    data1 = ('? 1', '+ 1', '? 1', '+ 2', 's 1 2', '+ 1000000000', '? 1000000000', '- 1000000000',
             '? 1000000000', 's 999999999 1000000000', '- 2', '? 2', '- 0', '+ 9', 's 0 9')
    ans1 = ('Not found', 'Found', '3', 'Found', 'Not found', '1', 'Not found', '10')

    data2 = ('? 0', '+ 0', '? 0', '- 0', '? 0')
    ans2 = ('Not found', 'Found', 'Not found')

    data3 = ('+ 491572259', '? 491572259', '? 899375874', 's 310971296 877523306', '+ 352411209')
    ans3 = ('Found', 'Not found', '491572259')

    assert check(*data1) == ans1
    assert check(*data2) == ans2
    assert check(*data3) == ans3

    assert check('+ 1', '+ 2', '+ 3', 's 1 3') == ('6',)
    assert check('s 1 10', '+ 5', 's 10 12', 's 1 10', 's 20 1') == ('0', '0', '5', '0')
    assert check('+ 4', 's 1 4', '- 4', 's 4 5') == ('4', '0')

    print('Everything is OK')

    n = 5 * 10**4
    tree = AVLTree()
    acc = 0
    for i in range(n):
        val = random.randint(0, 10**9)
        t0 = time.perf_counter()
        tree.insert(val)
        t1 = time.perf_counter()
        acc += (t1 - t0)
    for i in range(n):
        left_point = random.randint(0, 10**9)
        right_point = random.randint(left_point, 10**9)
        t0 = time.perf_counter()
        (res, tree) = find_sum_on_segment(tree, left_point, right_point)
        t1 = time.perf_counter()
        acc += (t1 - t0)
    
    print('Everything is OK. Execution time is {}'.format(acc))


def main():
    tree = AVLTree()
    n = int(input())
    assert 1 <= n <= 10**5

    for i in range(n):
        query, val = input().split(maxsplit=1)
        assert query in ('+', '-', '?', 's')
        if query in ('+', '-', '?'):
            val = int(val)
            assert 0 <= val <= 10**9

            if query == '+':
                tree.insert(val)
            elif query == '-':
                tree.delete(val)
            elif query == '?':
                print('Found' if tree.find(tree.root, val) else 'Not found')

        else:
            left_point, right_point = map(int, val.split())
            assert 0 <= left_point <= right_point <= 10**9

            (s, tree) = find_sum_on_segment(tree, left_point, right_point)
            print(s)

        # if tree.root.height:
            # tree.root.display()


def test2():
    tree = AVLTree()
    n = 20
    for i in range(n):
        tree = merge(tree, AVLTree(Node(i + 1, None, fake_node, fake_node, 1, i + 1)))
        print('{:-^72}'.format('tree'))
        tree.root.display()
        print('{:-^72}'.format('table'))
        tree.print_in_order()
        print('\n')


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, ValueError):
        print('Incorrect input')
    # test()
    # test2()
