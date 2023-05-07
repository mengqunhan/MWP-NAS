import torch
import numpy as np
import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# https://github.com/AlexZzander/PrefixToPostfix
class BinaryTreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.value)


class BinaryTree:
    def __init__(self):
        self.root = None


def prefix_to_postfix(expression):
    exlist_prefix = expression.split(" ")
    extree = constructree_fromprefix_expression(exlist_prefix)
    exlist_postfix = []
    post_traverse_extree(extree.root, exlist_postfix)
    return " ".join(exlist_postfix)


def post_traverse_extree(node, exlist2):
    if node is None:
        return
    post_traverse_extree(node.left, exlist2)
    post_traverse_extree(node.right, exlist2)
    exlist2.append(node.value)

def prefix_to_infix(expression):
    exlist_prefix = expression.split(" ")
    extree,construct_flag = constructree_fromprefix_expression(exlist_prefix)
    if not construct_flag:
        return ""
    exlist_infix = []
    in_traverse_extree(extree.root, exlist_infix)
    return "".join(exlist_infix)

def in_traverse_extree(node, exlist2):
    if node is None:
        return
    if node.left is not None:
        exlist2.append("(")
    in_traverse_extree(node.left, exlist2)
    exlist2.append(node.value)
    in_traverse_extree(node.right, exlist2)
    if node.right is not None:
        exlist2.append(")")


def constructree_fromprefix_expression(exlist):
    construct_flag=True
    operators = {'*', '/', '-', '+', '^'}
    stack = []
    tree = BinaryTree()
    for i in reversed(range(len(exlist))):
        if exlist[i] in operators:
            newnode = BinaryTreeNode(exlist[i])
            newnode.left = stack.pop()
            newnode.right = stack.pop()
            stack.append(newnode)
        else:
            stack.append(BinaryTreeNode(exlist[i]))
    tree.root = stack.pop()
    if len(stack)!=0:
        construct_flag=False
    return tree,construct_flag