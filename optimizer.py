import numpy as np


class Node:
    def __init__(self, level, choices, max_level):
        self.level = level
        self.max_level = max_level
        self.choices = choices
        self.children = []
        self.links = []
        self.total_children = len(choices)

    def expand(self):
        if self.level > self.max_level:
            print(self.level, self.max_level)
            return
        for choice in self.choices:
            choices = self.choices[:]
            choices.remove(choice)
            child = Node(self.level+1, choices, self.max_level)
            child.expand()
            self.total_children += child.total_children
            self.children.append(child)
            self.links.append(choice)

def explore(parent_node: Node):
    for child in parent_node.children:
        

def exhaustive_search(list1, list2):
    print(f"problem size={len(list1)*len(list2)}")
    assignment_matrix = np.zeros((len(list1), len(list2)))
    dim_x = len(list1)
    dim_y = len(list2)

    parent = Node(0, list2, len(list1))
    parent.expand()
    print(parent.total_children)
    return


if __name__ == '__main__':
    exhaustive_search([0, 1, 2, 3], [0, 1, 2, 3])
