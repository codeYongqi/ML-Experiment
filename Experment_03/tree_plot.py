# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

#用来正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
#用来正常显示符号
plt.rcParams['axes.unicode_minus'] = False


def tree_calculate(myTree):
    # 计算树的深度floor_num和每层的结点数node_num，并按层保存树的结点信息至myQueue
    # 输入参数
    #   myTree      构建的决策树的根结点，是字典类型，包含
    #                   label           该结点文本
    #                   class           该结点的类别标记
    #                   best            分支属性的id
    #                   branch_num      分支数
    #                   branchLabels    分支文本
    #                   branches        分支结点集合

    if len(myTree) == 0:
        return 0, []
    
    floor_num = 1
    node_num = [1]
    myQueue1 = [[myTree, -1]]       # myQueue1用于存放当前正在访问的层的所有结点、其父结点索引及其自身索引
    myQueue = [myQueue1]
    myQueue2 = []
    ii = 0                    # 当前正在访问的层次编号
    jj = 0                    # 当前即将访问的第ii层的结点的编号
    hasBranch = 0             # 当前已经访问的第ii层的前jj-1个结点是否有分支
    
    while jj < node_num[ii]:  # 依次访问第ii层的结点
        p = myQueue1[jj][0]
        
        if p["branch_num"] > 0:     # 当前结点有分支
            if hasBranch == 0:      # 当前已经访问的第ii层的前jj-1个结点没有分支
                hasBranch = 1
                floor_num = floor_num + 1                   # 新增一层
                node_num.append(p["branch_num"])            # 新增层次开始计算结点数
            else:                   # 当前已经访问的第ii层的前jj-1个结点已经有分支
                node_num[floor_num-1] += p["branch_num"]    # 第ii+1层结点数增加
            
            for tt in range(p["branch_num"]):    # 第ii+1层结点增加，放在myQueue2中
                myQueue2.append([p["branches"][tt], jj, tt])# jj是该结点的父结点在第ii层所有结点中的索引，tt是该结点在父结点的所有子结点中的索引
        
        if jj < node_num[ii]-1: # 第ii层还没有访问完
            jj += 1
        else:                   # 第ii层已经访问完
            if hasBranch == 1:  # 还有下一层
                ii += 1
                jj = 0
                hasBranch = 0
                myQueue1 = myQueue2
                myQueue.append(myQueue1)
                myQueue2 = []
            else:               # 没有下一层
                break

    return floor_num, node_num, myQueue


def tree_plot(myTree):
    # 绘制树的图
    # 输入参数
    #   myTree      构建的决策树的根结点，是字典类型，包含
    #                   label           该结点文本
    #                   class           该结点的类别标记
    #                   best            分支属性的id
    #                   branch_num      分支数
    #                   branchLabels    分支文本
    #                   branches        分支结点集合

    decisionNode = dict(boxstyle="square", fc="1.0")
    leafNode = dict(boxstyle="circle", fc="1.0")
    arrow_args = dict(arrowstyle="<|-")
    
    if len(myTree) == 0:
        print('一棵空树！')
        return
    
    # 计算树的深度floor_num和每层的结点数node_num，并按层获取树的结点信息至myQueue
    floor_num, node_num, myQueue = tree_calculate(myTree)

    # 设置所有结点坐标
    nodes_pos = [None] * floor_num
    gap_y = 1.0 / floor_num    # 结点之间垂直间距
    
    for ii in range(floor_num):
        nodes_pos[ii] = [None] * node_num[ii]
        gap_x = 1.0 / (node_num[ii] + 1)    # 结点之间水平间距
        
        for jj in range(node_num[ii]):
            nodes_pos[ii][jj] = ((jj+1) * gap_x, 1 - ii*gap_y)  # 结点x、y轴坐标

    # 开始绘制决策树，从最下面一层开始向上绘制（注：从上向下绘制显示效果会有点问题）
    fig = plt.figure(1, facecolor='white', dpi=600)    # 新建一个画布，背景设置为白色的
    fig.clf()                                          # 将画图清空
    ax = plt.subplot(111, frameon=False)               # 设置一个多图展示，但是子图只有一个
    
    for ii in range(floor_num-1, -1, -1):
        for jj in range(node_num[ii]):
            if myQueue[ii][jj][1] > -1:    # 当前结点有父结点
                if myQueue[ii][jj][0]["branch_num"] > 0:    # 当前结点有分支
                    ax.annotate(myQueue[ii][jj][0]['label'], xy=nodes_pos[ii-1][myQueue[ii][jj][1]], xycoords='axes fraction',
                                xytext=nodes_pos[ii][jj], textcoords='axes fraction', va="center", ha="center",
                                bbox=decisionNode, arrowprops=arrow_args)
                else:
                    ax.annotate(myQueue[ii][jj][0]['label'], xy=nodes_pos[ii-1][myQueue[ii][jj][1]], xycoords='axes fraction',
                                xytext=nodes_pos[ii][jj], textcoords='axes fraction', va="center", ha="center",
                                bbox=leafNode, arrowprops=arrow_args)

                # 绘制第ii层第jj个结点与其父结点之间的连线上的说明文字
                xMid = (nodes_pos[ii][jj][0] + nodes_pos[ii-1][myQueue[ii][jj][1]][0]) / 2
                yMid = (nodes_pos[ii][jj][1] + nodes_pos[ii-1][myQueue[ii][jj][1]][1]) / 2
                ax.text(xMid, yMid, myQueue[ii-1][myQueue[ii][jj][1]][0]["branchLabels"][myQueue[ii][jj][2]])
            else:    # 当前结点无父结点
                if myQueue[ii][jj][0]["branch_num"] > 0:    # 当前结点有分支
                    ax.annotate(myQueue[ii][jj][0]['label'], xy=nodes_pos[ii][jj], xycoords='axes fraction',
                                xytext=nodes_pos[ii][jj], textcoords='axes fraction', va="center", ha="center",
                                bbox=decisionNode, arrowprops=arrow_args)
                else:
                    ax.annotate(myQueue[ii][jj][0]['label'], xy=nodes_pos[ii][jj], xycoords='axes fraction',
                                xytext=nodes_pos[ii][jj], textcoords='axes fraction', va="center", ha="center",
                                bbox=leafNode, arrowprops=arrow_args)

    plt.show()