from collections import deque
import numpy as np

class TreeNode:
    def __init__(self,val,left,right):
        self.val = val
        self.left = left
        self.right = right
        self.par = None

    def is_leaf(self):
        return self.left == None and self.right == None

    def has_value(self):
        return self.val != None

def midorder(root):
    if root == None:
        return
    if root.left:
        midorder(root.left)
    print(root.val)
    if root.right:
        midorder(root.right)

def preorder(root):
    if root == None:
        return
    print(root.val)
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)

def traverse(root,target):
    if root == None:
        return -1
    if root.is_leaf() or root.left == None:
        return root.val
    if root.left.val >= target:
        return traverse(root.left,target)
    else:
        return traverse(root.right,target-root.left.val)


class SumTree:
    def __init__(self,input_list):
        self.leaf_list = deque()
        for i in range(len(input_list)):
            node = TreeNode(input_list[i],None,None)
            self.leaf_list.append(node)


        node_list = self.leaf_list.copy()
        k = 0
        while(len(node_list) > 1):
            i = 0
            l = int(len(node_list) / 2)
            #print(l)
            while(i < l):
                # calculate sum
                #print(i)
                sum = node_list[i].val + node_list[i+1].val
                # generate parent
                node = TreeNode(sum,node_list[i],node_list[i+1])
                if k == 0:
                    self.leaf_list[i].par = node
                    self.leaf_list[i+1].par = node
                else:
                    node_list[i].par = node
                    node_list[i+1].par = node

                node_list[i] = node
                node_list.remove(node_list[i+1])
                i += 1
            k += 1
        self.root = node_list[0]

    def print_tree1(self):
        midorder(self.root)

    def print_tree2(self):
        preorder(self.root)

    def traverse(self,target):
        return traverse(self.root,target)

    def update_tree(self,loc,val):
        self.leaf_list[loc].val = val
        p = self.leaf_list[loc]
        while(p.par != None):
            if p.par.left == p:
                p.par.val = p.par.right.val + p.val
            else:
                p.par.val = p.par.left.val + p.val
            p = p.par

    def pop_left(self):
        for i in range(1,len(self.leaf_list)):
            self.leaf_list[i-1].val = self.leaf_list[i].val

        self.leaf_list[-1].val = 0
        for j in range(len(self.leaf_list)):
            self.update_tree(j,self.leaf_list[j].val)