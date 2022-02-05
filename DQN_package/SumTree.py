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
        return 0
    if root.is_leaf() or root.left == None:
        return root.val
    if root.left.val >= target:
        return traverse(root.left,target)
    else:
        return traverse(root.right,target-root.left.val)

class SumTree:
    def __init__(self,input_list):
        leaf_list = []
        for i in range(len(input_list)):
            node = TreeNode(input_list[i],None,None)
            leaf_list.append(node)

        node_list = []
        while(len(leaf_list) > 1):
            i = 0
            l = int(len(leaf_list) / 2)
            #print(l)
            while(i < l):
                # calculate sum
                #print(i)
                sum = leaf_list[i].val + leaf_list[i+1].val
                # generate parent
                node = TreeNode(sum,leaf_list[i],leaf_list[i+1])
                leaf_list[i].par = node
                leaf_list[i+1].par = node

                node_list.append(leaf_list[i])
                node_list.append(leaf_list[i+1])

                leaf_list[i] = node
                leaf_list.remove(leaf_list[i+1])
                i += 1

        self.root = leaf_list[0]

    def print_tree1(self):
        midorder(self.root)

    def print_tree2(self):
        preorder(self.root)

    def traverse(self,target):
        return traverse(self.root,target)