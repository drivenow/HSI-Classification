# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 22:40:07 2015

@author: Shenjunling
"""

# coding=utf-8

__author__ = 'hujiawei'
__doc__ = 'pagerank mapreduce'

class Node:
    def __init__(self,id,pk):
        self.id=id
        self.pk=pk

def pk_map(map_input):
    map_output={}
    for node,outlinks in map_input.items():
        for link in outlinks:
            size=len(outlinks)
            if link in map_output:
                map_output[link]+=(float)(node.pk)/size
            else:
                map_output[link]=(float)(node.pk)/size
    return map_output

def pk_reduce(reduce_input):
    for result in reduce_input:
        for node,value in result.items():
            node.pk+=value

def pk_clear(nodes):
    for node in nodes:
        node.pk=0

def pk_last(nodes):
    lastnodes=[]
    for node in nodes:
        lastnodes.append(Node(node.id,node.pk))
    return lastnodes

def pk_diff(nodes,lastnodes):
    diff=0
    for i in range(len(nodes)):
        print('node pk %f, last node pk %f ' % (nodes[i].pk, lastnodes[i].pk))
        diff+=abs(nodes[i].pk-lastnodes[i].pk)
    return diff

def pk_test1():
    node1 = Node(1, 0.25)
    node2 = Node(2, 0.25)
    node3 = Node(3, 0.25)
    node4 = Node(4, 0.25)
    nodes = [node1, node2, node3, node4]
    threshold = 0.0001
    max_iters = 100

    for iter_count in range(max_iters):
        iter_count += 1
        lastnodes=pk_last(nodes)
        print('============ map count %d =================' % (iter_count))
        in1 = {node1: [node2, node3, node4], node2: [node3, node4]}
        in2 = {node3: [node1, node4], node4: [node2]}

        mapout1 = pk_map(in1)
        mapout2 = pk_map(in2)

        for node, value in mapout1.items():
            print str(node.id) + ' ' + str(value)

        for node, value in mapout2.items():
            print str(node.id) + ' ' + str(value)

        print('============ reduce count %d =================' % (iter_count))

        reducein = [mapout1, mapout2]
        pk_clear(nodes)
        pk_reduce(reducein)

        for node in nodes:
            print str(node.id) + ' ' + str(node.pk)

        diff=pk_diff(nodes,lastnodes)
        if diff < threshold:
            break

if __name__ == '__main__':
    pk_test1()
