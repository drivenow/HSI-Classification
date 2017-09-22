# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 15:56:31 2017

@author: Shenjunling
"""

import os 
import logging 
import subprocess
import sys
sys.path.append("G:/OneDrive/codes/python/DataLoad")
 
log = logging.getLogger("Core.Analysis.Processing") 
 
INTERPRETER = "F:/Anaconda2/envs/py3/python.exe" 
 
 
if not os.path.exists(INTERPRETER): 
  log.error("Cannot find INTERPRETER at path \"%s\"." % INTERPRETER) 
   
processor = "HSI_ResNet.py" 
batch_size = [25]
block_size = [13]
repeat = 10

for i in batch_size:
    for j in block_size:
        for re in range(repeat):       
            pargs = [INTERPRETER, processor] 
            pargs.append("-batch_size")
            pargs.append(str(i))
            pargs.append("-block_size")
            pargs.append(str(j))
            pargs.append("-test_size")
            pargs.append(str(0.99))
            print("===============================================================")
            print(pargs)
            p = subprocess.Popen(pargs)
            p.wait()