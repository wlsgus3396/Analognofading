import copy
import torch
import numpy as np
from torch import nn
import math

def amp(w_compressed, A, m):
    r = copy.deepcopy(w_compressed)
    iterAMP = 10**5
    s = np.zeros((len(w_compressed),1))
    sqm=math.sqrt(m)
    
    #A = [ [np.random.normal() for i in range(m)]    for j in range(len(w_compressed)) ] 
    #A = np.array(A)
    #A = A / (math.sqrt(len(w_compressed))) 
    
    for i in range(iterAMP):
        r=abs(r)
        r.sort()
        maxs=r.reverse()
        sigma=mean(maxs[0:19])
        g = np.transpose(A) * r
        s = s + g

        for j in range(len(w_compressed)):
            s[j] = np.sign(s[j]) * np.max(np.abs(s[j])-sigma, 0)
        
        #print(s)
        
        b = sum(np.abs(s)>0) / m
        r = w_compressed-A*s+b*r
    
    return s

     