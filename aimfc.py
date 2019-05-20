# -*- coding: utf-8 -*-
""" aimfc.py即目标函数，本例通过输入每一代的染色体，由自定义评价函数计算与目标曲线的面积差，作为目标函数值ObjV来输出 """

import numpy as np

def MakeObjCurve(width):
    ''' 创建目标曲线，此处定义为一组正弦波拼接序列 '''
    n1 = width//3
    n2 = width - 2*n1
    x1=np.cos(np.arange(0,n1)) * 1
    x2=np.cos(np.arange(0,n1)) * 4
    x3=np.cos(np.arange(0,n2)) * 2
    ObjCurve=np.hstack((x1,x2,x3))
    return ObjCurve

def CalScore(chrom):
    ''' 返回染色体与目标曲线之间的面积的倒数作为评分值 '''
    objCurve = MakeObjCurve(len(chrom))
    area = chrom - objCurve
    area *= 10**5                   #调整系数确保分值不受小数项干扰
    score = 1 / np.dot(area, area)  #计算差值的平方和以简化求面积过程
    return score

def myEvaFunc(chroms):
    ''' 自定义评价函数，以评分值作为目标函数值 '''
    scores = []
    for chrom in chroms:
        score = CalScore(chrom)
        scores.append(score)
    scores = np.array([scores]).T
    return scores

def aimfuc(Phen, LegV):

    ObjV = myEvaFunc(Phen)
    exIdx = np.argmin(ObjV[:, 0])

    # 惩罚方法2： 标记非可行解在可行性列向量中对应的值为0，并编写punishing罚函数来修改非可行解的适应度。
    # 也可以不写punishing，因为Geatpy内置的算法模板及内核已经对LegV标记为0的个体的适应度作出了修改。
    # 使用punishing罚函数实质上是对非可行解个体的适应度作进一步的修改
    LegV[exIdx] = 0 # 对非可行解作出标记，使其在可行性列向量中对应的值为0，此处标记的是得分最小项

    return [ObjV, LegV]