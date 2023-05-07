import torch
import re
from  sympy import simplify, expand
import sys
from collections import deque
from graphviz import Digraph
from random import sample
import uuid
from queue import Queue
import copy

class ETree:

    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None
        self.dot = Digraph(comment='Binary Tree')

    def negative_pass_down(self):

        operators = ['+', '*', '*-', '+/']
        if self.data not in operators:

            assert self.left == self.right is None
            if self.data[0] == '-':
                self.data = self.data[1:]
            else:
                self.data = '-' + self.data
        elif self.data == '+':

            self.left.negative_pass_down()
            self.right.negative_pass_down()
        elif self.data == '*':
            self.data = '*-'
        elif self.data == '*-':
            self.data = '*'
        else:

            self.left.negative_pass_down()
            self.right.negative_pass_down()

    def opposite_pass_down(self):

        operators = ['+', '*', '*-', '+/']
        if self.data not in operators:

            assert self.left == self.right is None

            if self.data[-1] == '/':
                self.data = self.data[:-1]
            else:
                self.data = self.data + '/'

        elif self.data == '+':
            self.data = '+/'

        elif self.data == '*':
            self.left.opposite_pass_down()
            self.right.opposite_pass_down()

        elif self.data == '*-':
            self.left.opposite_pass_down()
            self.right.opposite_pass_down()
        else:
            self.data = '+'

    def preorder(self):

        if self.data is not None:
            print(self.data, end=' ')
        if self.left is not None:
            self.left.preorder()
        if self.right is not None:
            self.right.preorder()

    def inorder(self):

        if self.left is not None:
            self.left.inorder()
        if self.data is not None:
            print(self.data, end=' ')
        if self.right is not None:
            self.right.inorder()

    def postorder(self):

        if self.left is not None:
            self.left.postorder()
        if self.right is not None:
            self.right.postorder()
        if self.data is not None:
            print(self.data, end=' ')

    def levelorder(self):

        def LChild_Of_Node(node):
            return node.left if node.left is not None else None

        def RChild_Of_Node(node):
            return node.right if node.right is not None else None

        level_order = []

        if self.data is not None:
            level_order.append([self])

        height = self.height()
        if height >= 1:

            for _ in range(2, height + 1):
                level = []
                for node in level_order[-1]:

                    if LChild_Of_Node(node):
                        level.append(LChild_Of_Node(node))

                    if RChild_Of_Node(node):
                        level.append(RChild_Of_Node(node))

                if level:
                    level_order.append(level)

            for i in range(0, height):
                for index in range(len(level_order[i])):
                    level_order[i][index] = level_order[i][index].data

        return level_order

    def height(self):

        if self.data is None:
            return 0
        elif self.left is None and self.right is None:
            return 1
        elif self.left is None and self.right is not None:
            return 1 + self.right.height()
        elif self.left is not None and self.right is None:
            return 1 + self.left.height()
        else:
            return 1 + max(self.left.height(), self.right.height())

    def leaves(self):
        leaves_count = 0
        if self.data is None:
            return None
        elif self.left is None and self.right is None:
            print(self.data, end=' ')
        elif self.left is None and self.right is not None:
            self.right.leaves()
        elif self.right is None and self.left is not None:
            self.left.leaves()
        else:
            self.left.leaves()
            self.right.leaves()

    def print_tree(self, save_path='./Binary_Tree.gv', label=False):

        colors = ['skyblue', 'tomato', 'orange', 'purple', 'green', 'yellow', 'pink', 'red']

        def print_node(node, node_tag):

            color = sample(colors, 1)[0]
            if node.left is not None:
                left_tag = str(uuid.uuid1())
                self.dot.node(left_tag, str(node.left.data), style='filled', color=color)
                label_string = 'L' if label else ''
                self.dot.edge(node_tag, left_tag, label=label_string)
                print_node(node.left, left_tag)

            if node.right is not None:
                right_tag = str(uuid.uuid1())
                self.dot.node(right_tag, str(node.right.data), style='filled', color=color)
                label_string = 'R' if label else ''
                self.dot.edge(node_tag, right_tag, label=label_string)
                print_node(node.right, right_tag)

        if self.data is not None:
            root_tag = str(uuid.uuid1())
            self.dot.node(root_tag, str(self.data), style='filled', color=sample(colors, 1)[0])
            print_node(self, root_tag)

class METree:
    def __init__(self, value: str, dot=None):
        self.data = value
        self.children = []
        self.parent = None
        if dot is None:
            self.dot = Digraph(comment='M-ary Tree')
        else:
            self.dot = dot
        self.label = None

    def print_tree_levelorder(self):
        q = Queue(maxsize=0)
        q.put(self)
        level_list = []
        while not q.empty():
            tt = q.get()

            if tt.parent is not None:
                print(tt.data, tt.parent.data, len(tt.children))
            else:
                print(tt.data, tt.parent, len(tt.children))

            if len(tt.children) > 0:
                for child in tt.children:
                    q.put(child)

    def recoding_levelorder(self):
        q = Queue(maxsize=0)
        q.put(self)
        q.put('stop!!!')

        level_list = []
        level_count = 0
        level_list.append([])

        while not q.empty():
            tt = q.get()
            if tt == 'stop!!!':
                level_count += 1
                level_list.append([])
                continue

            if tt.data in ['+', '*', '+/', '*-']:
                if tt.data not in level_list[level_count]:
                    level_list[level_count].append(tt.data)
                else:
                    tt.data = tt.data + '@'

                    while tt.data in level_list[level_count]:
                        tt.data = tt.data + '@'
                    level_list[level_count].append(tt.data)
            if len(tt.children) > 0:
                for child in tt.children:
                    q.put(child)
                q.put('stop!!!')

    def get_leaves(self) -> list:
        if self.data != '+':
            print('wrong! root is not + !!!!!')
            return []
        if len(self.children) == 0:
            print('wrong! NO child !!!!!')
            return []
        leaves = []
        q = Queue(maxsize=0)
        q.put(self)
        while not q.empty():
            tt = q.get()
            if tt.data not in ['+', '*', '+/', '*-'] and tt.data.find('@') == -1:
                leaves.append(tt)
            if len(tt.children) > 0:
                for child in tt.children:
                    q.put(child)
        return leaves

    def print_tree(self, save_path=None, label=False):

        colors = ['skyblue', 'tomato', 'orange', 'purple', 'green', 'yellow', 'pink', 'red']

        def print_node(node, node_tag):

            color = sample(colors, 1)[0]
            if len(node.children) > 0:
                for ccc in node.children:
                    child_tag = str(uuid.uuid1())
                    self.dot.node(child_tag, str(ccc.data), style='filled', color=color)
                    label_string = 'L' if label else ''
                    self.dot.edge(node_tag, child_tag, label=label_string)
                    print_node(ccc, child_tag)

        if self.data is not None:
            root_tag = str(uuid.uuid1())
            self.dot.node(root_tag, str(self.data), style='filled', color=sample(colors, 1)[0])
            print_node(self, root_tag)
        if save_path is not None:
            self.dot.render(save_path)

def clean_equation(raw_equation):
    '''
    raw_equation: a string of equation eg:'x=(11-1)*2'
    '''
    equation = re.sub(' ', '', raw_equation)    #去除空格

    equation = re.sub('（', '(', equation)
    equation = re.sub('）', ')', equation)

    equation = equation.replace('[', '(')
    equation = equation.replace(']', ')')

    equation = equation.replace(':', '/')

    equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
    equation = re.sub('(\d+)_\((\d+/\d+)\)', '(\\1+\\2)', equation)
    equation = re.sub('(\d+)_(\d+/\d+)', '(\\1+\\2)', equation)
    equation = re.sub('(\d+)\(', '\\1+(', equation)

    equation = re.sub('(\d+)\+\((\d+/\d+)\)', '\\1+\\2', equation)

    equation = re.sub('(\d+(,\d+)?(\.\d+)?)%', '(\\1/100)', equation)

    if equation[:2] == 'x=':
        equation = equation[2:]

    equation = equation.replace('^', '**')
    return 'x=' + equation

def substitute_equation(equation):
    '''
    equation:经过clean_equation后的equation
    '''

def simplify_ex(T_equation):
    if T_equation[:2] == 'x=':
        T_equation = T_equation[2:]
    new_ex = simplify(T_equation)

    new_ex = expand(new_ex)
    new_ex = str(new_ex)

    T_equation = new_ex.replace(' ', '')
    T_equation = T_equation.replace('^', '**')

    indd = T_equation.find('**')
    count = 0
    while indd != -1:
        count += 1
        if count > 20:
            T_equation = T_equation.replace('**', '^')
            return False, T_equation

        e_num = T_equation[indd + 2]
        if not e_num.isdigit():
            break
        num = re.split('\+|-|\*|/|\(',T_equation[:indd])[-1]
        # num = T_equation[indd - 1]

        if num == ')':

            s_begin = T_equation.rfind('(', 0, indd - 1)
            num = T_equation[s_begin:indd]

            sub_str = num
            for ii in range(int(e_num) - 1):
                sub_str += ('*' + num)

            if T_equation[s_begin - 1] == '/' and s_begin > 0:
                T_equation = T_equation[:s_begin] + '(' + sub_str + ')' + T_equation[indd + 3:]
            else:
                T_equation = T_equation[:s_begin] + sub_str + T_equation[indd + 3:]
            indd = T_equation.find('**')

        else:
            sub_str = num
            for ii in range(int(e_num) - 1):
                sub_str += ('*' + num)
            if T_equation[indd - 1 - len(num)] == '/' and indd > 1:
                T_equation = T_equation[:indd - len(num)] + '(' + sub_str + ')' + T_equation[indd + 3:]
            else:
                T_equation = T_equation[:indd - len(num)] + sub_str + T_equation[indd + 3:]
            indd = T_equation.find('**')

    return True, T_equation

def transform_ex_str_to_list(ex:str,index2word:list,num_start):
    ex_list = []
    splits_list = re.split(r'([+\-*/^()])', ex)
    for ch in splits_list:
        if ch != '':
            ex_list.append(ch)
    new_ex_list = []
    for idd, ch in enumerate(ex_list):
        if idd < 2:
            new_ex_list.append(ch)
        else:
            last_ch = ex_list[idd - 2]
            last_ch2 = ex_list[idd - 1]
            if last_ch == '(' and last_ch2 == '-' and ch in index2word[num_start:]:
                del new_ex_list[-1]
                new_ex_list.append('-' + ch)
            else:
                new_ex_list.append(ch)
    return new_ex_list

def infix_to_postfix(infix_input: list) -> list:
    """
    Converts infix expression to postfix.
    Args:
        infix_input(list): infix expression user entered
    """
    precedence_order = {'+': 0, '-': 0, '*': 1, '/': 1, '^': 2}
    associativity = {'+': "LR", '-': "LR", '*': "LR", '/': "LR", '^': "RL"}

    clean_infix = infix_input

    i = 0
    postfix = []
    operators = "+-/*^"
    stack = deque()
    while i < len(clean_infix):

        char = clean_infix[i]

        if char in operators:

            if len(stack) == 0 or stack[0] == '(':

                stack.appendleft(char)
                i += 1

            else:

                top_element = stack[0]

                if precedence_order[char] == precedence_order[top_element]:

                    if associativity[char] == "LR":

                        popped_element = stack.popleft()
                        postfix.append(popped_element)

                    elif associativity[char] == "RL":

                        stack.appendleft(char)
                        i += 1
                elif precedence_order[char] > precedence_order[top_element]:

                    stack.appendleft(char)
                    i += 1
                elif precedence_order[char] < precedence_order[top_element]:

                    popped_element = stack.popleft()
                    postfix.append(popped_element)
        elif char == '(':

            stack.appendleft(char)
            i += 1
        elif char == ')':
            top_element = stack[0]
            while top_element != '(':
                popped_element = stack.popleft()
                postfix.append(popped_element)

                top_element = stack[0]

            stack.popleft()
            i += 1

        else:
            postfix.append(char)
            i += 1

    if len(stack) > 0:
        for i in range(len(stack)):
            postfix.append(stack.popleft())

    return postfix

def construct_my_exp_tree(postfix: list):
    stack = []
    if '^' in postfix:
        print('Having ^ operator ！！！')
        return None
    for char in postfix:
        if char not in ["+", "-", "*", "/"]:
            t = ETree(char)
            stack.append(t)
        else:
            if char == '+' or char == '*':

                t = ETree(char)
                t1 = stack.pop()
                t2 = stack.pop()

                t.right = t1
                t.left = t2

                stack.append(t)

            elif char == '-':
                t = ETree('+')
                t1 = stack.pop()
                t2 = stack.pop()
                if t1.data not in ['+', '*']:
                    if t1.data[0] == '-':
                        t1.data = t1.data[1:]
                    else:
                        t1.data = '-' + t1.data
                else:
                    if t1.data == '+':
                        t1.negative_pass_down()
                    elif t1.data == '*':

                        t1.data = '*-'
                    else:
                        print('wrong 02: 出现了 + 和 * 外的内部节点！')
                        sys.exit()
                t.right = t1
                t.left = t2
                stack.append(t)

            elif char == '/':
                t = ETree('*')
                t1 = stack.pop()
                t2 = stack.pop()

                if t1.data not in ['+', '*']:
                    if t1.data[-1] == '/':
                        t1.data = t1.data[:-1]
                    else:
                        t1.data = t1.data + '/'
                else:
                    if t1.data == '+':
                        t1.data = '+/'
                    elif t1.data == '*':

                        t1.opposite_pass_down()
                t.right = t1
                t.left = t2
                stack.append(t)

            else:
                print('wrong 03: 后序表达式中出现了+-*/外的运算符号')
                sys.exit()

    t = stack.pop()
    return t

def construct_metree_from_betree_new(tree: ETree, parent: METree = None):
    node_data = tree.data
    parent_data = parent.data

    if node_data not in ['+', '*', '+/', '*-']:
        mtree = METree(tree.data)
        mtree.parent = parent
        parent.children.append(mtree)

        return mtree


    elif (parent_data == '+' and node_data == '+') or (parent_data == '+/' and node_data == '+') or (
            parent_data == '*' and node_data == '*') or (parent_data == '*-' and node_data == '*'):
        #node_data可以继续加到parent_data的孩子结点中，所以不需要新建结点
        construct_metree_from_betree_new(tree.left, parent) 
        construct_metree_from_betree_new(tree.right, parent)
    elif parent_data == '*' and node_data == '*-':
        parent.data = '*-'
        construct_metree_from_betree_new(tree.left, parent)
        construct_metree_from_betree_new(tree.right, parent)
    elif parent_data == '*-' and node_data == '*-':
        parent.data = '*'
        construct_metree_from_betree_new(tree.left, parent)
        construct_metree_from_betree_new(tree.right, parent)
    else:
        #node_data不能继续加到parent_data的孩子结点中，所以需要新建结点
        mtree = METree(tree.data)
        mtree.parent = parent

        construct_metree_from_betree_new(tree.left, mtree)
        construct_metree_from_betree_new(tree.right, mtree)
        parent.children.append(mtree)

        return mtree

def transform_path_to_label(key: str, path_ls: list):
    new_key = key
    if key[0] == '-':
        qufan = '1'
        new_key = new_key[1:]
    else:
        qufan = '0'

    if key[-1] == '/':
        qudaoshu = '1'
        new_key = new_key[:-1]
    else:
        qudaoshu = '0'

    code = '_'.join(path_ls)
    code = qufan + '_' + qudaoshu + '_' + code
    return new_key, code

def exp_to_mtree(exp,index2word,num_start):
    ccc_count = [0]
    ddone,expression = simplify_ex(exp)
    expression_list=transform_ex_str_to_list(expression,index2word,num_start)
    if expression_list[0] == '-':
        ccc_count[0] += 1
        expression_list = ['0'] + expression_list
    postfix_ex2=infix_to_postfix(expression_list)
    ex_tree=construct_my_exp_tree(postfix_ex2)
    MMMM=METree('+')
    construct_metree_from_betree_new(ex_tree,MMMM)
    MMMM.recoding_levelorder()

    my_num_codes={}
    leaves=MMMM.get_leaves()
    if len(leaves)>0:
        nums_labels={}
        for leaf in leaves:
            key=leaf.data
            path_to_root=[]
            parent = leaf.parent
            while parent is not None:
                path_to_root.append(parent.data)
                parent = parent.parent
            
            path_to_root.reverse()
            key,path_to_root=transform_path_to_label(key,path_to_root)

            if path_to_root not in my_num_codes.keys():
                my_num_codes[path_to_root] = 1
            else:
                my_num_codes[path_to_root] += 1

            if key in nums_labels.keys():
                nums_labels[key].append(path_to_root)
            else:
                nums_labels[key] = [path_to_root]
        return MMMM,nums_labels,my_num_codes
    else:
        print('NO leaves !!! WRONG !!!')
        nums_labels = None
        return MMMM, nums_labels, my_num_codes
    

def mtree_equal(mtree1,mtree2):
    if mtree1.data!=mtree2.data:
        return False
    else:
        if len(mtree1.children)==0 and len(mtree2.children)==0:
            return True
        elif len(mtree1.children)==len(mtree2.children):
            m1_children=mtree1.children
            m2_children=mtree2.children
            surplus_flag_m1=[1 for _ in range(len(m1_children))]
            surplus_flag_m2=[1 for _ in range(len(m2_children))]
            for i in range(len(m1_children)):
                if surplus_flag_m1[i]==1:
                    for j in range(len(m2_children)):
                        if surplus_flag_m2[j]==1:
                            if mtree_equal(m1_children[i],m2_children[j]):
                                surplus_flag_m1[i]=0
                                surplus_flag_m2[j]=0
                                break
            if sum(surplus_flag_m1)==0 and sum(surplus_flag_m2)==0:
                return True
            else:
                return False
        else:
            return False

def mtree_equal_code(nums_labels1,nums_labels2):
    labels1=copy.deepcopy(nums_labels1)
    labels2=copy.deepcopy(nums_labels2)
    new_label1={}
    new_label2={}
    for k,v in labels1.items():
        new_label1[k]=sorted(v)
    for k,v in labels2.items():
        new_label2[k]=sorted(v)
    if new_label1==new_label2:
        return True
    else:
        return False

def IoU(nums_labels1,nums_labels2):
    new_labels1=[]
    new_labels2=[]
    for k,v in nums_labels1.items():
        for c in v:
            new_labels1.append(k+c)
    for k,v in nums_labels2.items():
        for c in v:
            new_labels2.append(k+c)
    intersection=[]
    for l1 in new_labels1:
        for l2 in new_labels2:
            if l1==l2:
                intersection.append(l1)
    union=list(set(new_labels1+new_labels2))
    return len(intersection)/len(union)

def postfix_to_infix(postfix):
    rev_postfix=copy.deepcopy(postfix)
    stack = []
    rev_postfix.reverse()
    for token in rev_postfix:
        if token not in ['+', '-', '*', '/', '^']:
            stack.append(token)
        else:
            try:
                op1 = stack.pop()
                op2 = stack.pop()
            except:
                print(rev_postfix)
                exit()
            stack.append('({}{}{})'.format(op1, token, op2))
    if len(stack) != 1:
        print('error')
        exit()
    return stack[0]

def index2word_exp(exp,index2word):
    new_exp=[]
    for i in exp:
        if i <len(index2word):
            new_exp.append(index2word[i])
        else:
            # new_exp.append(i)
            assert False
    return new_exp

