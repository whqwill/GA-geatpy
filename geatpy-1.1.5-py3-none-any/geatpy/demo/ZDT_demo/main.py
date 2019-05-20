# -*- coding: utf-8 -*-
"""
执行脚本main.py
描述：
    该demo是展示如何利用Geatpy计算多目标优化标准测试函数ZDT1,2,3,4,6的帕累托前沿
    本案例通过调用moea_q_sorted_templet算法模板来解决该多目标优化问题，
    其详细用法可利用help命令查看，或是在github下载并查看源码
    其中目标函数ZDT1,2,3,4,6写在aimfuc.py文件中
"""

import numpy as np
import geatpy as ga # 导入geatpy库

# 获取函数接口地址
AIM_M = __import__('aimfuc')

"""============================变量设置============================"""
# for ZDT1-3
ranges = np.vstack([np.zeros((1,30)), np.ones((1,30))])   # 生成自变量的范围矩阵
borders = np.vstack([np.ones((1,30)), np.ones((1,30))])   # 生成自变量的边界矩阵
precisions = [4] * 30 # 在二进制/格雷码编码中代表自变量的编码精度，当控制变量是连续型时，根据crtfld参考资料，该变量只表示边界精度，故设置为一定的正数即可
AIM_F = 'ZDT1'

# for ZDT4
#ranges = np.vstack([-5 * np.ones((1,9)), 5 * np.ones((1,9))])   # 生成自变量的范围矩阵
#ranges = np.hstack([np.array([[0],[1]]), ranges])
#borders = np.vstack([np.ones((1,10)), np.ones((1,10))])   # 生成自变量的边界矩阵
#precisions = [4] * 10 # 在二进制/格雷码编码中代表自变量的编码精度，当控制变量是连续型时，根据crtfld参考资料，该变量只表示边界精度，故设置为一定的正数即可
#AIM_F = 'ZDT4'

# for ZDT6
#ranges = np.vstack([np.zeros((1,10)), 5 * np.ones((1,10))])   # 生成自变量的范围矩阵
#borders = np.vstack([np.ones((1,10)), np.ones((1,10))])   # 生成自变量的边界矩阵
#precisions = [4] * 10 # 在二进制/格雷码编码中代表自变量的编码精度，当控制变量是连续型时，根据crtfld参考资料，该变量只表示边界精度，故设置为一定的正数即可
#AIM_F = 'ZDT6'

"""========================遗传算法参数设置========================="""
NIND = 50                 # 种群规模
MAXGEN = 1000             # 最大遗传代数
GGAP = 1                  # 代沟：子代与父代的重复率为(1-GGAP),由于后面使用NSGA2算法，因此该参数无用
selectStyle = 'etour'     # 遗传算法的选择方式——带精英策略的锦标赛选择
recombinStyle = 'xovdprs' # 遗传算法的重组方式，设为使用代理的两点交叉
recopt = 0.9              # 交叉概率
pm = 0.1                  # 变异概率
SUBPOP = 1                # 设置种群数为1
maxormin = 1              # 设置标记表明这是最小化目标
MAXSIZE = 2000            # 帕累托最优解集中解的最大个数
FieldDR = ga.crtfld(ranges, borders, precisions) # 生成区域描述器
"""=======================调用编程模板进行种群进化==================="""
# 得到帕累托最优解集NDSet以及解集对应的目标函数值NDSetObjV
[ObjV, NDSet, NDSetObjV, times] = ga.moea_q_sorted_templet(AIM_M, AIM_F, None, None, FieldDR, 'R', maxormin, MAXGEN, MAXSIZE, NIND, SUBPOP, GGAP, selectStyle, recombinStyle, recopt, pm, distribute = True, drawing = 1)
