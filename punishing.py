# -*- coding: utf-8 -*-
"""punishing.py - 罚函数demo"""

import numpy as np

def punishing(LegV, FitnV):
    FitnV[np.where(LegV == 0)[0]] = 0
    return FitnV