# -*- coding: utf-8 -*-
""" main.py即主函数，本例仅用于演示“已知曲线寻优”的过程 """

import numpy as np
import geatpy as ga
import time
import matplotlib.pyplot as plt
import os

def search_objects(directory):
    directory=os.path.normpath(directory)  #规格化，防止分隔符造成的差异
    if not os.path.isdir(directory):
        raise IOError("The directory '"+"' doesn't exist!")
    objects={}
    for curdir,substrs,files in os.walk(directory):
        for jpeg in (file for file in files if file.endswith('.csv')):
            path=os.path.join(curdir,jpeg)
            label=path.split(os.path.sep)[-2]
            if label not in objects:
                objects[label]=[]
            objects[label].append(path)
    return objects

def sga_mps_real_templet(AIM_M, AIM_F, PUN_M, PUN_F, FieldDRs, problem, maxormin, MAXGEN, NIND, SUBPOP, GGAP, selectStyle, recombinStyle, recopt, pm, distribute, drawing = 1):
    """ 基于多种群独立进化单目标编程模板(实值编码)，各种群独立将父子两代合并进行选择，采取精英保留机制 """

    #==========================初始化配置===========================
    GGAP = 0.5 # 因为父子合并后选择，因此要将代沟设为0.5以维持种群规模
    # 获取目标函数和罚函数
    aimfuc = getattr(AIM_M, AIM_F) # 获得目标函数
    if PUN_F is not None:
        punishing = getattr(PUN_M, PUN_F) # 获得罚函数
    NVAR = FieldDRs[0].shape[1] # 得到控制变量的个数
    # 定义全局进化记录器，初始值为nan
    pop_trace = (np.zeros((MAXGEN ,2)) * np.nan)
    pop_trace[:, 0] = 0
    # 定义变量记录器，记录控制变量值，初始值为nan
    var_trace = (np.zeros((MAXGEN ,NVAR)) * np.nan)
    """=========================开始遗传算法进化======================="""
    start_time = time.time() # 开始计时
    # 对于各个网格分别进行进化，采用全局进化记录器记录最优值
    for index in range(len(FieldDRs)): # 遍历各个子种群，各子种群独立进化，互相不竞争
        FieldDR = FieldDRs[index]
        if problem == 'R':
            Chrom = ga.crtrp(NIND, FieldDR) # 生成初始种群
        elif problem == 'I':
            Chrom = ga.crtip(NIND, FieldDR)
        LegV = np.ones((NIND, 1)) # 初始化种群的可行性列向量
        [ObjV, LegV] = aimfuc(Chrom, LegV) # 求初始种群的目标函数值
        repnum = 0 # 初始化重复个体数为0
        ax = None # 存储上一帧图形
        gen = 0
        badCounter = 0 # 用于记录在“遗忘策略下”被忽略的代数
        # 开始进化！！
        while gen < MAXGEN:
            if badCounter >= 10 * MAXGEN: # 若多花了10倍的迭代次数仍没有可行解出现，则跳出
                break
            # 进行遗传算子，生成子代
            SelCh = ga.recombin(recombinStyle, Chrom, recopt, SUBPOP) # 重组
            if problem == 'R':
                SelCh = ga.mutbga(SelCh,FieldDR, pm) # 变异
                if repnum > Chrom.shape[0] * 0.01: # 当最优个体重复率高达1%时，进行一次高斯变异
                    SelCh = ga.mutgau(SelCh, FieldDR, pm) # 高斯变异
            elif problem == 'I':
                SelCh = ga.mutint(SelCh, FieldDR, pm)
            LegVSel = np.ones((SelCh.shape[0], 1)) # 初始化育种种群的可行性列向量
            [ObjVSel, LegVSel] = aimfuc(SelCh, LegVSel) # 求育种种群的目标函数值
            # 父子合并
            Chrom = np.vstack([Chrom, SelCh])
            ObjV = np.vstack([ObjV, ObjVSel])
            LegV = np.vstack([LegV, LegVSel])
            FitnV = ga.ranking(maxormin * ObjV, LegV, None, SUBPOP) # 适应度评价
            if PUN_F is not None:
                FitnV = punishing(LegV, FitnV) # 调用惩罚函数
            repnum = len(np.where(ObjV[np.argmax(FitnV)] == ObjV)[0]) # 计算最优个体重复数
            # 记录进化过程
            bestIdx = np.argmax(FitnV)
            if (LegV[bestIdx] != 0) and ((np.isnan(pop_trace[gen,1])) or ((maxormin == 1) & (pop_trace[gen,1] >= ObjV[bestIdx])) or ((maxormin == -1) & (pop_trace[gen,1] <= ObjV[bestIdx]))):
                feasible = np.where(LegV != 0)[0] # 排除非可行解
                pop_trace[gen,0] += np.sum(ObjV[feasible]) / ObjV[feasible].shape[0] / len(FieldDRs) # 记录种群个体平均目标函数值
                pop_trace[gen,1] = ObjV[bestIdx] # 记录当代目标函数的最优值
                var_trace[gen,:] = Chrom[bestIdx, :] # 记录当代最优的控制变量值
                # 绘制动态图
                if drawing == 2:
                    ax = ga.sgaplot(pop_trace[:,[1]],'子种群'+str(index+1)+'各代种群最优个体目标函数值', False, ax, gen)
                badCounter = 0 # badCounter计数器清零
            else:
                gen -= 1 # 忽略这一代（遗忘策略）
                badCounter += 1
            if distribute == True: # 若要增强种群的分布性（可能会造成收敛慢）
                idx = np.argsort(ObjV[:, 0], 0)
                dis = np.diff(ObjV[idx,0]) / (np.max(ObjV[idx,0]) - np.min(ObjV[idx,0]) + 1)# 差分计算距离的修正偏移量
                dis = np.hstack([dis, dis[-1]])
                dis = dis + np.min(dis) # 修正偏移量+最小量=修正绝对量
                FitnV[idx, 0] *= np.exp(dis) # 根据相邻距离修改适应度，突出相邻距离大的个体，以增加种群的多样性
            [Chrom, ObjV, LegV] = ga.selecting(selectStyle, Chrom, FitnV, GGAP, SUBPOP, ObjV, LegV) # 选择
            gen += 1
    end_time = time.time() # 结束计时
    times = end_time - start_time
    # 后处理进化记录器
    delIdx = np.where(np.isnan(pop_trace))[0]
    pop_trace = np.delete(pop_trace, delIdx, 0)
    var_trace = np.delete(var_trace, delIdx, 0)
    if pop_trace.shape[0] == 0:
        raise RuntimeError('error: no feasible solution. (有效进化代数为0，没找到可行解。)')
    # 输出结果
    if maxormin == 1:
        best_gen = np.argmin(pop_trace[:, 1]) # 记录最优种群是在哪一代
        best_ObjV = np.min(pop_trace[:, 1])
    elif maxormin == -1:
        best_gen = np.argmax(pop_trace[:, 1]) # 记录最优种群是在哪一代
        best_ObjV = np.max(pop_trace[:, 1])
    print('最优的目标函数值为：%s'%(best_ObjV))
    print('最优的控制变量值为：')
    for i in range(NVAR):
        print(var_trace[best_gen, i])
    print('有效进化代数：%s'%(pop_trace.shape[0]))
    print('最优的一代是第 %s 代'%(best_gen + 1))
    print('时间已过 %s 秒'%(times))
    # 绘图
    if drawing != 0:
        ga.trcplot(pop_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])
    # 返回进化记录器、变量记录器以及执行时间
    return [pop_trace, var_trace, times, best_gen]

# 获取函数接口地址
AIM_M = __import__('aimfc')
PUN_M = __import__('punishing')
POP_SIZE = 300          # 种群高度
CHROM_LENGTH = 20       # 染色体宽度
max_generation = 150    # 进化代数
chrom_bottom = -4       #染色体数值下限
chrom_top = 4           #染色体数值上限

# 变量设置
x = []; b = []
for i in range(CHROM_LENGTH):
    x.append([chrom_bottom, chrom_top]) # 自变量的范围
    b.append([0, 0]) # 自变量是否包含下界
ranges=np.vstack(x).T # 生成自变量的范围矩阵
borders = np.vstack(b).T # 生成自变量的边界矩阵
precisions = [1]*CHROM_LENGTH # 在二进制/格雷码编码中代表自变量的编码精度，当控制变量是连续型时，根据crtfld参考资料，该变量只表示边界精度，故设置为一定的正数即可
# 生成网格化后的区域描述器集合
FieldDRs = []
for i in range(1):
    FieldDRs.append(ga.crtfld(ranges, borders, precisions))

# 调用编程模板(设置problem = 'R'处理实数型变量问题，详见该算法模板的源代码)
[pop_trace, var_trace, times, best_gen] = sga_mps_real_templet(AIM_M, 'aimfuc', PUN_M, 'punishing',
 FieldDRs, problem = 'R', maxormin = -1, MAXGEN = max_generation, NIND = POP_SIZE, SUBPOP = 1, GGAP = 0.9, \
 selectStyle = 'tour', recombinStyle = 'xovdprs', recopt = 0.9, pm = 0.3, distribute = True, drawing = 1)

bstChrom = var_trace[best_gen]
objCurve = AIM_M.MakeObjCurve(CHROM_LENGTH)

plt.ion()
fig = plt.figure('曲线寻优演示',facecolor='lightgray')
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.set_title("Evaluation Map")
ax1.grid(axis='y', linestyle=':')
for i in range(max_generation):
    if i%5==0:
        ax1.plot(var_trace[i], 'o-')
        ax2.cla()
        ax2.set_title("最优染色体[gen:%i]"%(i+1))
        ax2.plot(var_trace[i], 'o-', c='dodgerblue')
        plt.pause(0.001)
ax2.cla()
ax2.grid(axis='y', linestyle=':')
ax2.plot(objCurve, 'o-', c='orangered', label='目标曲线')
ax2.plot(bstChrom, 'o-', c='dodgerblue', label='最优染色体[gen:%i]'%(best_gen+1))
plt.legend()
plt.ioff()
plt.show()