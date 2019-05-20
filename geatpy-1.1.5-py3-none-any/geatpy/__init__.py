#    This file is part of Geatpy.
#
#    Geatpy is a free toolbox: you can redistribute it and/or modify
#    it as you want.
#
#    Geatpy is distributed in the hope that it will be useful for The Genetic 
#    and Evolutionary Algorithm, you can get the tutorial from www.geatpy.com
#
#    If you want to donate to it, please e-mail jazzbin@geatpy.com

__author__ = "Geatpy Team"
__version__ = "1.1.5"
__revision__ = "1.1.5"

# -*- coding: utf-8 -*-
"""
geatpy  -  import all libs of geatpy

Created on Sat May 12 08:36:49 2018

@author: jazzbin
"""

import sys
import platform

lib_path = __file__[:-11] + 'lib' + platform.architecture()[0][:2] + '/v' + sys.version[:3] + '/'
if lib_path not in sys.path:
    sys.path.append(lib_path)

from awGA import awGA
from bs2int import bs2int
from bs2rv import bs2rv
from crtbase import crtbase
from crtbp import crtbp
from crtfld import crtfld
from crtip import crtip
from crtpp import crtpp
from crtrp import crtrp
from etour import etour
from frontplot import frontplot
from indexing import indexing
from meshrng import meshrng
from migrate import migrate
from moea_awGA_templet import moea_awGA_templet
from moea_nsga2_templet import moea_nsga2_templet
from moea_q_sorted_new_templet import moea_q_sorted_new_templet
from moea_q_sorted_templet import moea_q_sorted_templet
from moea_rwGA_templet import moea_rwGA_templet
from mut import mut
from mutate import mutate
from mutbga import mutbga
from mutbin import mutbin
from mutgau import mutgau
from mutint import mutint
from mutpp import mutpp
from ndomindeb import ndomindeb
from ndomin import ndomin
from ndominfast import ndominfast
from powing import powing
from ranking import ranking
from recdis import recdis
from recint import recint
from reclin import reclin
from recombin import recombin
from redisNDSet import redisNDSet
from reins import reins
from rep import rep
from rwGA import rwGA
from rws import rws
from scaling import scaling
from selecting import selecting
from sga_code_templet import sga_code_templet
from sga_mpc_real_templet import sga_mpc_real_templet
from sga_mps_real_templet import sga_mps_real_templet
from sga_new_code_templet import sga_new_code_templet
from sga_permut_templet import sga_permut_templet
from sga_new_permut_templet import sga_new_permut_templet
from sga_real_templet import sga_real_templet
from sga_new_real_templet import sga_new_real_templet
from sgaplot import sgaplot
from sus import sus
from tour import tour
from trcplot import trcplot
from upNDSet import upNDSet
from xovdp import xovdp
from xovdprs import xovdprs
from xovmp import xovmp
from xovpm import xovpm
from xovsh import xovsh
from xovshrs import xovshrs
from xovsp import xovsp
from xovsprs import xovsprs
