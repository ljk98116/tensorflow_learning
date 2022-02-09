from DQN_package.SumTree import *

def test():
    l1 = [1,2,4,6,7]
    l2 = [1,3,5,6]
    l3 = [3,10,12,4,1,2,8,2]

    tree1 = SumTree(l1)
    tree2 = SumTree(l2)
    tree3 = SumTree(l3)

    tree1.print_tree1()
    print()
    tree1.print_tree2()
    print()
    tree2.print_tree1()
    print()
    tree2.print_tree2()
    print()
    tree3.print_tree1()
    print()
    tree3.print_tree2()
    print()
    print(tree3.traverse(24))
    print()
    for i in range(len(tree3.leaf_list)):
        print(tree3.leaf_list[i].val)
    print()
    tree1.update_tree(0,2)
    tree1.print_tree1()
    print()
    tree1.print_tree2()

if __name__ == "__main__":
    test()