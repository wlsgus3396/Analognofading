import copy
import torch
import numpy as np
from torch import nn
from models.flatten_params import flatten_params
from models.recover_flattened import recover_flattened
import matplotlib
import matplotlib.pyplot as plt
import time
import math
import sympy
from sympy import Integral, Symbol, exp, S, sqrt
import ipdb

import os
import sys

vp_path = os.path.abspath(__file__)
sys.path.append(__file__)

from  models import vampyre as vp
from csvec import CSVec

import operator as op
from functools import reduce



""""###########################################################################################"""
""""###########################################################################################"""
def FedAvg(w,device):
    model=[]

    for k in w[0].keys():
        model.append(w[0][k].shape)


    # W에 w[0] 유저의 모든 파라미터 값을 리스트로 저장한다.
    W=[]

    for k in w[0].keys():
        W.append(w[0][k])

    D,L=flatten_params(np.array(W))
    d=len(D)
    # D, L = flatten_params(np.array(W))
    # 리스트를 플래튼 시킨다.
    # D= 플래튼 값 21840개, tensor
    # L= 인덱스
    

    Fw=torch.zeros((len(w),d)).to(device)
    # shape: user수 * 21840

    for i in range(0,len(w)):
        W=[]
        for k in w[0].keys():
            W.append(w[i][k])
        # W에  모든 파라미터를 저장

        D,L=flatten_params(np.array(W))

        D=D.reshape(21840)
        Fw[i]=D

        # for k in range(0,d):
        #     Fw[i][k]=D[k]
        #     #Fw에 각 유저의 파라미터를 순차적으로 저장



    Fw_avg = copy.deepcopy(D)

    Fw_avg[:]=0
    Fw_avg=Fw.sum(axis=0)
    Fw_avg=Fw_avg/len(w)


    # for k in range(0, d):
    #     Fw_avg[k]=0
    #     # Fw_avg 값 초기화
    #     for i in range(0, len(w)):
    #         Fw_avg[k] += Fw[i][k]
    #     Fw_avg[k] = Fw_avg[k]/len(w)
    #     # 더해서 평균 구하기

    W_avg = recover_flattened(Fw_avg,L,model)
    # 파라미터 형태로 복원


    w_avg=copy.deepcopy(w[0])
    j=0
    for k in w_avg.keys():
        w_avg[k]=W_avg[j]
        j+=1

    #w_avg에 넣기


    return w_avg

""""###########################################################################################"""
""""###########################################################################################"""





""""###########################################################################################"""
""""###########################################################################################"""
def FedSketch(w, previous_w_global,T,P,K,cc, g_ec,s_momentum,po,add_momentum,device,iter,column,add_error):
    # 모델의 shape 저장하기 "conv1.weight", "conv1.bias" ... ... ... 등

    model = []
    for k in w[0].keys():
        model.append(w[0][k].shape)

    # W에 모든 파라미터 값을 리스트로 저장한다.
    W = []
    for k in w[0].keys():
        W.append(w[0][k])

    D, L = flatten_params(W)
    # D, L = flatten_params(np.array(W))
    # D= 플래튼 값
    # L= 인덱스

    d = len(D)

    ##############################################################
    G = torch.zeros((len(w), d)).to(device)  # [유저 * 플래튼 값]
    OG = torch.zeros((len(w), d)).to(device)
    ##############################################################


    """새로 업데이트된 웨이트 값 플래튼 시키기"""
    Fw = torch.zeros((len(w), d)).to(device)

    for i in range(0, len(w)):
        W = []
        for k in w[0].keys():
            W.append(w[i][k])
        D, L = flatten_params(W)
        D=D.reshape(21840)
        Fw[i]=D


    """ 직전의 글로벌 웨이트 값 플래튼 시키기"""
    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])
    D, L = flatten_params((W))


    """ Gradient 구하기"""
    D=D.reshape(21840)
    for i in range(len(w)):
        G[i]=(D-Fw[i])/0.01 +g_ec[i]
        OG[i]=G[i]

    dd = d
    ccrr=2*T  # 10000 / 2000
    rr = int(ccrr/cc)   # 25    /   10



    # 각 유저별 스케칭 준비
    Sketch_0 = CSVec(dd, cc, rr, device=device)
    Sketch_1 = CSVec(dd, cc, rr, device=device)
    Sketch_2 = CSVec(dd, cc, rr, device=device)
    Sketch_3 = CSVec(dd, cc, rr, device=device)
    Sketch_4 = CSVec(dd, cc, rr, device=device)
    Sketch_5 = CSVec(dd, cc, rr, device=device)
    Sketch_6 = CSVec(dd, cc, rr, device=device)
    Sketch_7 = CSVec(dd, cc, rr, device=device)
    Sketch_8 = CSVec(dd, cc, rr, device=device)
    Sketch_9 = CSVec(dd, cc, rr, device=device)

    # 인원수 늘리기

    Sketch_10 = CSVec(dd, cc, rr, device=device)
    Sketch_11 = CSVec(dd, cc, rr, device=device)
    Sketch_12 = CSVec(dd, cc, rr, device=device)
    Sketch_13 = CSVec(dd, cc, rr, device=device)
    Sketch_14 = CSVec(dd, cc, rr, device=device)
    Sketch_15 = CSVec(dd, cc, rr, device=device)
    Sketch_16 = CSVec(dd, cc, rr, device=device)
    Sketch_17 = CSVec(dd, cc, rr, device=device)
    Sketch_18 = CSVec(dd, cc, rr, device=device)
    Sketch_19 = CSVec(dd, cc, rr, device=device)
    Sketch_20 = CSVec(dd, cc, rr, device=device)
    Sketch_21 = CSVec(dd, cc, rr, device=device)
    Sketch_22 = CSVec(dd, cc, rr, device=device)
    Sketch_23 = CSVec(dd, cc, rr, device=device)
    Sketch_24 = CSVec(dd, cc, rr, device=device)
    Sketch_25 = CSVec(dd, cc, rr, device=device)
    Sketch_26 = CSVec(dd, cc, rr, device=device)
    Sketch_27 = CSVec(dd, cc, rr, device=device)
    Sketch_28 = CSVec(dd, cc, rr, device=device)
    Sketch_29 = CSVec(dd, cc, rr, device=device)
    Sketch_30 = CSVec(dd, cc, rr, device=device)
    Sketch_31 = CSVec(dd, cc, rr, device=device)
    Sketch_32 = CSVec(dd, cc, rr, device=device)
    Sketch_33 = CSVec(dd, cc, rr, device=device)
    Sketch_34 = CSVec(dd, cc, rr, device=device)
    Sketch_35 = CSVec(dd, cc, rr, device=device)
    Sketch_36 = CSVec(dd, cc, rr, device=device)
    Sketch_37 = CSVec(dd, cc, rr, device=device)
    Sketch_38 = CSVec(dd, cc, rr, device=device)
    Sketch_39 = CSVec(dd, cc, rr, device=device)
    Sketch_40 = CSVec(dd, cc, rr, device=device)
    Sketch_41 = CSVec(dd, cc, rr, device=device)
    Sketch_42 = CSVec(dd, cc, rr, device=device)
    Sketch_43 = CSVec(dd, cc, rr, device=device)
    Sketch_44 = CSVec(dd, cc, rr, device=device)
    Sketch_45 = CSVec(dd, cc, rr, device=device)
    Sketch_46 = CSVec(dd, cc, rr, device=device)
    Sketch_47 = CSVec(dd, cc, rr, device=device)
    Sketch_48 = CSVec(dd, cc, rr, device=device)
    Sketch_49 = CSVec(dd, cc, rr, device=device)
    Sketch_50 = CSVec(dd, cc, rr, device=device)
    Sketch_51 = CSVec(dd, cc, rr, device=device)
    Sketch_52 = CSVec(dd, cc, rr, device=device)
    Sketch_53 = CSVec(dd, cc, rr, device=device)
    Sketch_54 = CSVec(dd, cc, rr, device=device)
    Sketch_55 = CSVec(dd, cc, rr, device=device)
    Sketch_56 = CSVec(dd, cc, rr, device=device)
    Sketch_57 = CSVec(dd, cc, rr, device=device)
    Sketch_58 = CSVec(dd, cc, rr, device=device)
    Sketch_59 = CSVec(dd, cc, rr, device=device)
    Sketch_60 = CSVec(dd, cc, rr, device=device)
    Sketch_61 = CSVec(dd, cc, rr, device=device)
    Sketch_62 = CSVec(dd, cc, rr, device=device)
    Sketch_63 = CSVec(dd, cc, rr, device=device)
    Sketch_64 = CSVec(dd, cc, rr, device=device)
    Sketch_65 = CSVec(dd, cc, rr, device=device)
    Sketch_66 = CSVec(dd, cc, rr, device=device)
    Sketch_67 = CSVec(dd, cc, rr, device=device)
    Sketch_68 = CSVec(dd, cc, rr, device=device)
    Sketch_69 = CSVec(dd, cc, rr, device=device)
    Sketch_70 = CSVec(dd, cc, rr, device=device)
    Sketch_71 = CSVec(dd, cc, rr, device=device)
    Sketch_72 = CSVec(dd, cc, rr, device=device)
    Sketch_73 = CSVec(dd, cc, rr, device=device)
    Sketch_74 = CSVec(dd, cc, rr, device=device)
    Sketch_75 = CSVec(dd, cc, rr, device=device)
    Sketch_76 = CSVec(dd, cc, rr, device=device)
    Sketch_77 = CSVec(dd, cc, rr, device=device)
    Sketch_78 = CSVec(dd, cc, rr, device=device)
    Sketch_79 = CSVec(dd, cc, rr, device=device)
    Sketch_80 = CSVec(dd, cc, rr, device=device)
    Sketch_81 = CSVec(dd, cc, rr, device=device)
    Sketch_82 = CSVec(dd, cc, rr, device=device)
    Sketch_83 = CSVec(dd, cc, rr, device=device)
    Sketch_84 = CSVec(dd, cc, rr, device=device)
    Sketch_85 = CSVec(dd, cc, rr, device=device)
    Sketch_86 = CSVec(dd, cc, rr, device=device)
    Sketch_87 = CSVec(dd, cc, rr, device=device)
    Sketch_88 = CSVec(dd, cc, rr, device=device)
    Sketch_89 = CSVec(dd, cc, rr, device=device)
    Sketch_90 = CSVec(dd, cc, rr, device=device)
    Sketch_91 = CSVec(dd, cc, rr, device=device)
    Sketch_92 = CSVec(dd, cc, rr, device=device)
    Sketch_93 = CSVec(dd, cc, rr, device=device)
    Sketch_94 = CSVec(dd, cc, rr, device=device)
    Sketch_95 = CSVec(dd, cc, rr, device=device)
    Sketch_96 = CSVec(dd, cc, rr, device=device)
    Sketch_97 = CSVec(dd, cc, rr, device=device)
    Sketch_98 = CSVec(dd, cc, rr, device=device)
    Sketch_99 = CSVec(dd, cc, rr, device=device)
    Sketch_100 = CSVec(dd, cc, rr, device=device)
    Sketch_101 = CSVec(dd, cc, rr, device=device)
    Sketch_102 = CSVec(dd, cc, rr, device=device)
    Sketch_103 = CSVec(dd, cc, rr, device=device)
    Sketch_104 = CSVec(dd, cc, rr, device=device)
    Sketch_105 = CSVec(dd, cc, rr, device=device)
    Sketch_106 = CSVec(dd, cc, rr, device=device)
    Sketch_107 = CSVec(dd, cc, rr, device=device)
    Sketch_108 = CSVec(dd, cc, rr, device=device)
    Sketch_109 = CSVec(dd, cc, rr, device=device)
    Sketch_110 = CSVec(dd, cc, rr, device=device)
    Sketch_111 = CSVec(dd, cc, rr, device=device)
    Sketch_112 = CSVec(dd, cc, rr, device=device)
    Sketch_113 = CSVec(dd, cc, rr, device=device)
    Sketch_114 = CSVec(dd, cc, rr, device=device)
    Sketch_115 = CSVec(dd, cc, rr, device=device)
    Sketch_116 = CSVec(dd, cc, rr, device=device)
    Sketch_117 = CSVec(dd, cc, rr, device=device)
    Sketch_118 = CSVec(dd, cc, rr, device=device)
    Sketch_119 = CSVec(dd, cc, rr, device=device)
    Sketch_120 = CSVec(dd, cc, rr, device=device)
    Sketch_121 = CSVec(dd, cc, rr, device=device)
    Sketch_122 = CSVec(dd, cc, rr, device=device)
    Sketch_123 = CSVec(dd, cc, rr, device=device)
    Sketch_124 = CSVec(dd, cc, rr, device=device)
    Sketch_125 = CSVec(dd, cc, rr, device=device)
    Sketch_126 = CSVec(dd, cc, rr, device=device)
    Sketch_127 = CSVec(dd, cc, rr, device=device)
    Sketch_128 = CSVec(dd, cc, rr, device=device)
    Sketch_129 = CSVec(dd, cc, rr, device=device)
    Sketch_130 = CSVec(dd, cc, rr, device=device)
    Sketch_131 = CSVec(dd, cc, rr, device=device)
    Sketch_132 = CSVec(dd, cc, rr, device=device)
    Sketch_133 = CSVec(dd, cc, rr, device=device)
    Sketch_134 = CSVec(dd, cc, rr, device=device)
    Sketch_135 = CSVec(dd, cc, rr, device=device)
    Sketch_136 = CSVec(dd, cc, rr, device=device)
    Sketch_137 = CSVec(dd, cc, rr, device=device)
    Sketch_138 = CSVec(dd, cc, rr, device=device)
    Sketch_139 = CSVec(dd, cc, rr, device=device)
    Sketch_140 = CSVec(dd, cc, rr, device=device)
    Sketch_141 = CSVec(dd, cc, rr, device=device)
    Sketch_142 = CSVec(dd, cc, rr, device=device)
    Sketch_143 = CSVec(dd, cc, rr, device=device)
    Sketch_144 = CSVec(dd, cc, rr, device=device)
    Sketch_145 = CSVec(dd, cc, rr, device=device)
    Sketch_146 = CSVec(dd, cc, rr, device=device)
    Sketch_147 = CSVec(dd, cc, rr, device=device)
    Sketch_148 = CSVec(dd, cc, rr, device=device)
    Sketch_149 = CSVec(dd, cc, rr, device=device)
    Sketch_150 = CSVec(dd, cc, rr, device=device)
    Sketch_151 = CSVec(dd, cc, rr, device=device)
    Sketch_152 = CSVec(dd, cc, rr, device=device)
    Sketch_153 = CSVec(dd, cc, rr, device=device)
    Sketch_154 = CSVec(dd, cc, rr, device=device)
    Sketch_155 = CSVec(dd, cc, rr, device=device)
    Sketch_156 = CSVec(dd, cc, rr, device=device)
    Sketch_157 = CSVec(dd, cc, rr, device=device)
    Sketch_158 = CSVec(dd, cc, rr, device=device)
    Sketch_159 = CSVec(dd, cc, rr, device=device)
    Sketch_160 = CSVec(dd, cc, rr, device=device)
    Sketch_161 = CSVec(dd, cc, rr, device=device)
    Sketch_162 = CSVec(dd, cc, rr, device=device)
    Sketch_163 = CSVec(dd, cc, rr, device=device)
    Sketch_164 = CSVec(dd, cc, rr, device=device)
    Sketch_165 = CSVec(dd, cc, rr, device=device)
    Sketch_166 = CSVec(dd, cc, rr, device=device)
    Sketch_167 = CSVec(dd, cc, rr, device=device)
    Sketch_168 = CSVec(dd, cc, rr, device=device)
    Sketch_169 = CSVec(dd, cc, rr, device=device)
    Sketch_170 = CSVec(dd, cc, rr, device=device)
    Sketch_171 = CSVec(dd, cc, rr, device=device)
    Sketch_172 = CSVec(dd, cc, rr, device=device)
    Sketch_173 = CSVec(dd, cc, rr, device=device)
    Sketch_174 = CSVec(dd, cc, rr, device=device)
    Sketch_175 = CSVec(dd, cc, rr, device=device)
    Sketch_176 = CSVec(dd, cc, rr, device=device)
    Sketch_177 = CSVec(dd, cc, rr, device=device)
    Sketch_178 = CSVec(dd, cc, rr, device=device)
    Sketch_179 = CSVec(dd, cc, rr, device=device)
    Sketch_180 = CSVec(dd, cc, rr, device=device)
    Sketch_181 = CSVec(dd, cc, rr, device=device)
    Sketch_182 = CSVec(dd, cc, rr, device=device)
    Sketch_183 = CSVec(dd, cc, rr, device=device)
    Sketch_184 = CSVec(dd, cc, rr, device=device)
    Sketch_185 = CSVec(dd, cc, rr, device=device)
    Sketch_186 = CSVec(dd, cc, rr, device=device)
    Sketch_187 = CSVec(dd, cc, rr, device=device)
    Sketch_188 = CSVec(dd, cc, rr, device=device)
    Sketch_189 = CSVec(dd, cc, rr, device=device)
    Sketch_190 = CSVec(dd, cc, rr, device=device)
    Sketch_191 = CSVec(dd, cc, rr, device=device)
    Sketch_192 = CSVec(dd, cc, rr, device=device)
    Sketch_193 = CSVec(dd, cc, rr, device=device)
    Sketch_194 = CSVec(dd, cc, rr, device=device)
    Sketch_195 = CSVec(dd, cc, rr, device=device)
    Sketch_196 = CSVec(dd, cc, rr, device=device)
    Sketch_197 = CSVec(dd, cc, rr, device=device)
    Sketch_198 = CSVec(dd, cc, rr, device=device)
    Sketch_199 = CSVec(dd, cc, rr, device=device)
    Sketch_200 = CSVec(dd, cc, rr, device=device)
    Sketch_201 = CSVec(dd, cc, rr, device=device)
    Sketch_202 = CSVec(dd, cc, rr, device=device)
    Sketch_203 = CSVec(dd, cc, rr, device=device)
    Sketch_204 = CSVec(dd, cc, rr, device=device)
    Sketch_205 = CSVec(dd, cc, rr, device=device)
    Sketch_206 = CSVec(dd, cc, rr, device=device)
    Sketch_207 = CSVec(dd, cc, rr, device=device)
    Sketch_208 = CSVec(dd, cc, rr, device=device)
    Sketch_209 = CSVec(dd, cc, rr, device=device)
    Sketch_210 = CSVec(dd, cc, rr, device=device)
    Sketch_211 = CSVec(dd, cc, rr, device=device)
    Sketch_212 = CSVec(dd, cc, rr, device=device)
    Sketch_213 = CSVec(dd, cc, rr, device=device)
    Sketch_214 = CSVec(dd, cc, rr, device=device)
    Sketch_215 = CSVec(dd, cc, rr, device=device)
    Sketch_216 = CSVec(dd, cc, rr, device=device)
    Sketch_217 = CSVec(dd, cc, rr, device=device)
    Sketch_218 = CSVec(dd, cc, rr, device=device)
    Sketch_219 = CSVec(dd, cc, rr, device=device)
    Sketch_220 = CSVec(dd, cc, rr, device=device)
    Sketch_221 = CSVec(dd, cc, rr, device=device)
    Sketch_222 = CSVec(dd, cc, rr, device=device)
    Sketch_223 = CSVec(dd, cc, rr, device=device)
    Sketch_224 = CSVec(dd, cc, rr, device=device)
    Sketch_225 = CSVec(dd, cc, rr, device=device)
    Sketch_226 = CSVec(dd, cc, rr, device=device)
    Sketch_227 = CSVec(dd, cc, rr, device=device)
    Sketch_228 = CSVec(dd, cc, rr, device=device)
    Sketch_229 = CSVec(dd, cc, rr, device=device)
    Sketch_230 = CSVec(dd, cc, rr, device=device)
    Sketch_231 = CSVec(dd, cc, rr, device=device)
    Sketch_232 = CSVec(dd, cc, rr, device=device)
    Sketch_233 = CSVec(dd, cc, rr, device=device)
    Sketch_234 = CSVec(dd, cc, rr, device=device)
    Sketch_235 = CSVec(dd, cc, rr, device=device)
    Sketch_236 = CSVec(dd, cc, rr, device=device)
    Sketch_237 = CSVec(dd, cc, rr, device=device)
    Sketch_238 = CSVec(dd, cc, rr, device=device)
    Sketch_239 = CSVec(dd, cc, rr, device=device)
    Sketch_240 = CSVec(dd, cc, rr, device=device)
    Sketch_241 = CSVec(dd, cc, rr, device=device)
    Sketch_242 = CSVec(dd, cc, rr, device=device)
    Sketch_243 = CSVec(dd, cc, rr, device=device)
    Sketch_244 = CSVec(dd, cc, rr, device=device)
    Sketch_245 = CSVec(dd, cc, rr, device=device)
    Sketch_246 = CSVec(dd, cc, rr, device=device)
    Sketch_247 = CSVec(dd, cc, rr, device=device)
    Sketch_248 = CSVec(dd, cc, rr, device=device)
    Sketch_249 = CSVec(dd, cc, rr, device=device)




    Sketch = [Sketch_0, Sketch_1, Sketch_2, Sketch_3, Sketch_4, Sketch_5, Sketch_6, Sketch_7, Sketch_8, Sketch_9,
              Sketch_10, Sketch_11, Sketch_12, Sketch_13, Sketch_14, Sketch_15, Sketch_16, Sketch_17, Sketch_18, Sketch_19,
              Sketch_20, Sketch_21, Sketch_22, Sketch_23, Sketch_24, Sketch_25, Sketch_26, Sketch_27, Sketch_28, Sketch_29,
              Sketch_30, Sketch_31, Sketch_32, Sketch_33, Sketch_34, Sketch_35, Sketch_36, Sketch_37, Sketch_38, Sketch_39,
              Sketch_40, Sketch_41, Sketch_42, Sketch_43, Sketch_44, Sketch_45, Sketch_46, Sketch_47, Sketch_48, Sketch_49,
              Sketch_50, Sketch_51, Sketch_52, Sketch_53, Sketch_54, Sketch_55, Sketch_56, Sketch_57, Sketch_58, Sketch_59,
              Sketch_60, Sketch_61, Sketch_62, Sketch_63, Sketch_64, Sketch_65, Sketch_66, Sketch_67, Sketch_68, Sketch_69,
              Sketch_70, Sketch_71, Sketch_72, Sketch_73, Sketch_74, Sketch_75, Sketch_76, Sketch_77, Sketch_78, Sketch_79,
              Sketch_80, Sketch_81, Sketch_82, Sketch_83, Sketch_84, Sketch_85, Sketch_86, Sketch_87, Sketch_88, Sketch_89,
              Sketch_90, Sketch_91, Sketch_92, Sketch_93, Sketch_94, Sketch_95, Sketch_96, Sketch_97, Sketch_98, Sketch_99,
              Sketch_100, Sketch_101, Sketch_102, Sketch_103, Sketch_104, Sketch_105, Sketch_106, Sketch_107, Sketch_108, Sketch_109,
              Sketch_110, Sketch_111, Sketch_112, Sketch_113, Sketch_114, Sketch_115, Sketch_116, Sketch_117, Sketch_118, Sketch_119,
              Sketch_120, Sketch_121, Sketch_122, Sketch_123, Sketch_124, Sketch_125, Sketch_126, Sketch_127, Sketch_128, Sketch_129,
              Sketch_130, Sketch_131, Sketch_132, Sketch_133, Sketch_134, Sketch_135, Sketch_136, Sketch_137, Sketch_138, Sketch_139,
              Sketch_140, Sketch_141, Sketch_142, Sketch_143, Sketch_144, Sketch_145, Sketch_146, Sketch_147, Sketch_148, Sketch_149,
              Sketch_150, Sketch_151, Sketch_152, Sketch_153, Sketch_154, Sketch_155, Sketch_156, Sketch_157, Sketch_158, Sketch_159,
              Sketch_160, Sketch_161, Sketch_162, Sketch_163, Sketch_164, Sketch_165, Sketch_166, Sketch_167, Sketch_168, Sketch_169,
              Sketch_170, Sketch_171, Sketch_172, Sketch_173, Sketch_174, Sketch_175, Sketch_176, Sketch_177, Sketch_178, Sketch_179,
              Sketch_180, Sketch_181, Sketch_182, Sketch_183, Sketch_184, Sketch_185, Sketch_186, Sketch_187, Sketch_188, Sketch_189,
              Sketch_190, Sketch_191, Sketch_192, Sketch_193, Sketch_194, Sketch_195, Sketch_196, Sketch_197, Sketch_198, Sketch_199,
              Sketch_200, Sketch_201, Sketch_202, Sketch_203, Sketch_204, Sketch_205, Sketch_206, Sketch_207, Sketch_208, Sketch_209,
              Sketch_210, Sketch_211, Sketch_212, Sketch_213, Sketch_214, Sketch_215, Sketch_216, Sketch_217, Sketch_218, Sketch_219,
              Sketch_220, Sketch_221, Sketch_222, Sketch_223, Sketch_224, Sketch_225, Sketch_226, Sketch_227, Sketch_228, Sketch_229,
              Sketch_230, Sketch_231, Sketch_232, Sketch_233, Sketch_234, Sketch_235, Sketch_236, Sketch_237, Sketch_238, Sketch_239,
              Sketch_240, Sketch_241, Sketch_242, Sketch_243, Sketch_244, Sketch_245, Sketch_246, Sketch_247, Sketch_248, Sketch_249]


    """스케칭  x.table에 스케칭한 값이 들어있음."""
    for i in range(len(w)):
        #Sketch[i].accumulateVec(Tensor_OG[i])
        Sketch[i].accumulateVec(OG[i])
        # EX) Sketch[0].table.shape= 100 by 100

    """채널 없이 해보기 - 시작 -"""
    # sum_sketch=torch.zeros([rr,cc]).to(device)
    # for i in range(len(w)):
    #
    #     sum_sketch+=Sketch[i].table
    #
    # #평균값
    # sum_sketch=sum_sketch/len(w)
    # noise = ((torch.randn(rr, cc).to(device)))/len(w)
    #
    # #sum_sketch +=noise
    #
    # """채널 없이 해보기 -끝-"""
########################################################################################################################
########################################################################################################################
    
    """타임슬롯 및 ofdm 적용방식"""
    # sub carrier= 50개
    # Time slot= 100개 가정
    h2_matrix={}     #  100 by 50 매트릭스
    s_matrix = {}   # 100 by 50 매트릭스
    sign_matrix = {} # 100 by 100 매트릭스

    # exponential 샘플링
    sampling = torch.distributions.exponential.Exponential(1.0)


    for m in range(len(w)):
        # h^2의 매트릭스 구하기 1*T SIZE
        h2_matrix[m] = sampling.sample([1, T])

        # h^2 > 0.001 적용, 조건보다 작은값은 0으로 만들기
        s_matrix[m]=h2_matrix[m]>0.001
        vector=np.vectorize(np.int) # 브로드캐스팅을 위한 함수 설정
        s_matrix[m]=vector(s_matrix[m]) # 1,0으로 표시

        s_matrix[m]=torch.tensor(s_matrix[m]).to(device) # 텐서 변환

        # s_matrix 복제해서 가로로 연결하기: 같은 페이딩 겪게 하기 (1*2T)
        #sign_matrix[m]=np.concatenate([s_matrix[m],s_matrix[m]], axis=1)
        sign_matrix[m]=torch.cat([s_matrix[m],s_matrix[m]], axis=1)

    # 넘파이 -> 텐서
    #for i in range(len(w)):
    #    sign_matrix[i]=torch.tensor(sign_matrix[i]).to(device)

    """# sign_matrix의 합 구하기: 차후 나눠주기 위해서..식(14) """

    sum_sign_matrix=torch.zeros(1,2*T).to(device)
    for m in range(len(w)):
        sum_sign_matrix += sign_matrix[m]



    """# E_1(X) 함수 만들기: 근사치로 사용"""
    x=Symbol('x')
    f=exp(-x)/x
    # E_1(0.001)로 하기
    E_1=Integral(f,(x,0.001,S.Infinity)).doit().evalf()
    # .doit() 실행 메소드
    # .evalf() 숫자로 나오게 하기

    """근사치"""
    # E_1 근사치: 0.001일때, 6.33153936413615
    # E_1 근사치: 0.005일때, 4.726095458584443


    gradient_sum_timeslot = {}
    sketch_to_vector={}

    ###################################################################################################################
    """Power allocation: 찬호논문 버전"""
    # for m in range(len(w)):
    #     # sketchmatrix(rr,cc) -> vector(1,2T)
    #
    #     sketch_to_vector[m]=((Sketch[m].table).reshape(1,2*T))
    #
    #     # 제곱 -> (1,2T) -> (2,T) 만들어서 더하기 -> (1,T) -> (100,50) 변형 -> 가로로 더함. (100,1) 은 타임슬롯별 그래디언트 제곱합
    #     # 예) n=1, 그래디언트 100개의 제곱 합
    #     gradient_sum_timeslot[m]=((((sketch_to_vector[m]**2).reshape(2,T)).sum(axis=0)).reshape(int(T/50),50)).sum(axis=1)

    """Power allocation: 진현논문 버전"""
    for m in range(len(w)):
        sketch_to_vector[m] = ((Sketch[m].table).reshape(1, 2 * T))
        gradient_sum_timeslot[m] = sum(((((sketch_to_vector[m]**2).reshape(2, T)).sum(axis=0)).reshape(int(T / 50), 50)).sum(axis=1))


    """"# gamma 구하기 식(20): 찬호버전"""
    gamma=[]

    # for m in range(len(w)):
    #     for i in range(int(T/50)): # Number of timeslot
    #         #gamma.append((torch.sqrt(sampling.sample([1]).to(device))) * (torch.sqrt(po / gradient_sum_timeslot[m][i]) / np.sqrt(6.33153936413615)))
    #         gamma.append(torch.sqrt(po / gradient_sum_timeslot[m][i]) / np.sqrt(6.33153936413615))


    """"# gamma 구하기 식(20): 진현버전"""
    for m in range(len(w)):
        for i in range(int(T / 50)):
            gamma.append(torch.sqrt(T/50*po / gradient_sum_timeslot[m]) / np.sqrt(6.33153936413615))


    # gamma list -> gamma matrix로 변형
    gamma_matrix=torch.tensor(gamma).reshape(len(w),int(T/50)).to(device)

    # gamma_t를 타임슬롯별로 복사하기
    # time slot n= 1,      2,  ....       100
    # gamma_t   [1-50]  [51-100]     [4950-5000]  각 타임슬롯별 동일 감마값 적용
    # 5000size의 감마벡터를 두개를 가로로 연결하여 10000size 벡터로 만들기

    gamma_vector_cat={}
    for m in range(len(w)):
        gamma_vector_T=torch.zeros([int(T/50),50]).to(device)
        for r in range(int(T/50)):
            # 각행에 같은 gamma 값 넣기,
            gamma_vector_T[r]=gamma_matrix[m][r]
        # 같은 감마값 (50),(50) ...(50).  50개 세트가 100개있음
        # 1*T 사이즈로 변경후 가로로 연결
        gamma_vector=gamma_vector_T.reshape(1,T)
        gamma_vector_cat[m]=torch.cat([gamma_vector,gamma_vector], axis=1)


    """# gamma 평균구하기: 식 (21)"""
    gamma_avg=torch.zeros(1,2*T).to(device)
    for m in range(len(w)):
        gamma_avg += gamma_vector_cat[m]
    gamma_avg=gamma_avg/len(w)
    ###################################################################################################################


    """# Received signal 구하기: 식(22)"""
    re_signal=torch.zeros(1,2*T).to(device)
    for m in range(len(w)):
        re_signal+=gamma_vector_cat[m]*sketch_to_vector[m]*sign_matrix[m]


    # 가우시안 노이즈 더하기
    noise= ((torch.randn(1, 2 * T).to(device)) / np.sqrt(2))


    #re_signal_noise= re_signal+noise
    """# 감마값 나눠주고, 평균구하기: 식(23) 식(24)"""
    #signal_avg= re_signal_noise/sum_sign_matrix
    #scdown_signal=signal_avg/(gamma_avg)


    """시그널과 노이즈 분리 계산 """
    signal_avg=re_signal/sum_sign_matrix
    scdown_signal=signal_avg/gamma_avg

    noise_avg= noise/sum_sign_matrix
    scdown_noise= noise_avg/gamma_avg

    final_signal=scdown_signal+scdown_noise


    # E ="\n".join(map(str, torch.sort(scdown_signal)[0][0]))
    # f = open('./result/scdown_signal_P{}_c{}_iter{}.csv'.format(po,column,iter),'w')
    # f.write(E)
    # f.close()
    #
    # Q ="\n".join(map(str, torch.sort(final_signal)[0][0]))
    # f = open('./result/final_signalP{}_c{}_iter{}.csv'.format(po,column,iter),'w')
    # f.write(Q)
    # f.close()



    # 노이즈 없이 실험해보기
    #final_signal=scdown_signal

    #ipdb.set_trace()

    # vec_to_matrix
    re_matrix=final_signal.reshape(rr,cc)



    #########################################################################################
    """기존 방식"""
    ##########################################################################################
    """ 모멘텀 주기: 동일 실험 조건을 위해 모멘텀 빼기"""

    if add_momentum:
        s_momentum = 0.9 * s_momentum + re_matrix

    else:
        s_momentum=re_matrix


    """ 모든 값 복원하기"""
    Tensor_matrix = s_momentum # 다시 텐서로 변환

########################################################################################################################
########################################################################################################################

    """채널없이 해보기"""
    # Tensor_matrix=sum_sketch


    vals = torch.zeros(rr, dd).to(device)
    for r in range(rr):
        vals[r] = (Tensor_matrix[r, Sketch_0.buckets[r, :]] * Sketch_0.signs[r, :])
    #ipdb.set_trace()

    """rr이 짝수이면, median 2개의값에 대한 평균값 적용, 홀수이면 median값"""
    if rr%2==0:
        recoverd_g=torch.quantile(vals, q=torch.tensor([0.5]).to(device), dim=0)[0]
    else:
        recoverd_g=torch.median(vals,dim=0)[0]

    #recoverd_g = vals.median(dim=0)[0]



    # 에러 누적하기
    recoverd_g += g_ec


    """ Top-k 적용"""
    return_topk = torch.zeros_like(recoverd_g)
    _, topkIndices = torch.topk(recoverd_g ** 2, k=K)  #



    return_topk[topkIndices] = recoverd_g[topkIndices]

    """ error accumulation"""
    # 새로운 에러 누적시키기
    # error_accumulation=Recovery_G - return_topk

    # 뺄셈대신, 0으로 만들기
    recoverd_g[topkIndices] = 0

    
    # 에러 어큐물 실행 여부
    if add_error:
        g_ec=recoverd_g

    # recoverd_g to CSV

    # A ="\n".join(map(str, torch.sort(return_topk)[0]))
    # f = open('./result/gradient_P{}_c{}_iter{}.csv'.format(po,column,iter),'w')
    # f.write(A)
    # f.close()



    """weight 업데이트"""
    Fw_avg = copy.deepcopy(D)
    Fw_avg -=0.01 * return_topk

    # Fw_avg = copy.deepcopy(D)
    # for k in range(0, d):
    #     Fw_avg[k] -= 0.01 * return_topk[k]

    """ flatten -> 매트릭스"""
    W_avg = recover_flattened(Fw_avg, L, model)

    """ 모델 매트릭스에 값 넣기"""
    w_avg = copy.deepcopy(w[0])
    j = 0
    for k in w_avg.keys():
        w_avg[k] = W_avg[j]
        j += 1

    return w_avg, g_ec, s_momentum



""""###########################################################################################"""
""""###########################################################################################"""


""""###########################################################################################"""
""""###########################################################################################"""


def FedAnalogRandom(w, previous_w_global, T, P, g_ec,po,device,add_error):
    model = []
    for k in w[0].keys():
        model.append(w[0][k].shape)

    W = []
    for k in w[0].keys():
        W.append(w[0][k])
    D, L = flatten_params(W)

    d = len(D)


    ##############################################################
    G = torch.zeros((len(w), d)).to(device)
    OG = torch.zeros((len(w), d)).to(device)
    #Sum = torch.zeros(len(w)).to(device)
    restored_gradient = torch.zeros(d).to(device)
    ##############################################################

    "새로 업데이트된 웨이트 값 플래튼 시키기"
    Fw = torch.zeros((len(w), d)).to(device)

    for i in range(len(w)):
        W = []
        for k in w[0].keys():
            W.append(w[i][k])

        D, L = flatten_params(W)

        D=D.reshape(21840)
        Fw[i]=D

        #for k in range(0, d):
        #    Fw[i][k] = D[k]

    "직전의 글로벌 웨이트 값 플래튼 시키기"
    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])
    D, L = flatten_params(W)

    "그래디언트 구하기"
    D=D.reshape(21840)
    for i in range(len(w)):
        G[i] = (D - Fw[i])/0.01 + g_ec[i]
        OG[i] = G[i]


    ###############################################################

    "파라미터 중 2T개 랜덤하게 선택하기 "
    Sp=np.random.choice(d,2*T,replace=False)
    Sp=torch.tensor(Sp).to(device)
    Sp=torch.sort(Sp)[0]


    "랜덤하게 선택한 곳에 1 넣기"
    SP = torch.zeros(d).to(device)
    SP[Sp]=1


    "유저별 패턴위치 값 추출하고 error 보존 "

    compressed_gradient=torch.zeros(len(w),2*T).to(device)
    pattern_gradient=torch.zeros(len(w),d).to(device)
    for i in range(len(w)):
        # 2T개만 추출하기
        compressed_gradient[i]=OG[i][Sp]
        # 에러보존하기
        pattern_gradient[i]=OG[i]*SP


        # 에러 보존 여부
        if add_error:
            g_ec[i]=OG[i]-pattern_gradient[i]



    """#########################채널 환경 구성하기#######################################"""

    h2_matrix = {}  # 50 by 100 매트릭스
    s_matrix = {}  # 50 by 100 매트릭스
    sign_matrix = {}  # 100 by 100 매트릭스

    sampling = torch.distributions.exponential.Exponential(1.0)

    for m in range(len(w)):
        h2_matrix[m] = sampling.sample([1, T])

        # h^2 > 0.001 적용, 조건보다 작은값은 0으로 만들기
        s_matrix[m] = h2_matrix[m] > 0.001
        vector = np.vectorize(np.int)  # 브로드캐스팅을 위한 함수 설정
        s_matrix[m] = vector(s_matrix[m])  # 1,0으로 표시

        s_matrix[m] = torch.tensor(s_matrix[m]).to(device)  # 텐서 변환

        # s_matrix 복제해서 가로로 연결하기: 같은 페이딩 겪게 하기
        sign_matrix[m] = torch.cat([s_matrix[m], s_matrix[m]], axis=1)

    # sign_matrix의 합 구하기: 차후 나눠주기 위해서
    sum_sign_matrix = torch.zeros(1, 2 * T).to(device)
    for m in range(len(w)):
        sum_sign_matrix += sign_matrix[m]

    gradient_sum_timeslot = {}
    for m in range(len(w)):
        gradient_sum_timeslot[m] = sum((((((compressed_gradient[m])**2).reshape(2, T)).sum(axis=0)).reshape(int(T / 50), 50)).sum(axis=1))

    # E_1 구하기
    E_1 = 6.33153936413615
    # 4.726095458584443

    ####################################################################################################################
    # gamma 구하기
    gamma = []
    for m in range(len(w)):
        for i in range(int(T / 50)):
            gamma.append(torch.sqrt(T/50*po / gradient_sum_timeslot[m]) / np.sqrt(6.33153936413615))


    gamma_matrix = torch.tensor(gamma).reshape(len(w), int(T / 50)).to(device)

    # timeslot 구간 별 gamma 평균 구하기

    gamma_vector_cat = {}
    for m in range(len(w)):
        gamma_vector_T = torch.zeros([int(T / 50), 50]).to(device)
        for r in range(int(T / 50)):
            # 각행에 같은 gamma 값 넣기,
            gamma_vector_T[r] = gamma_matrix[m][r]
        # 같은 감마값 (50),(50) ...(50).  50개 세트가 100개있음
        # 1*T 사이즈로 변경후 가로로 연결
        gamma_vector = gamma_vector_T.reshape(1, T)
        gamma_vector_cat[m] = torch.cat([gamma_vector, gamma_vector], axis=1)

    """# gamma 평균구하기: 식 (21)"""
    gamma_avg = torch.zeros(1, 2 * T).to(device)
    for m in range(len(w)):
        gamma_avg += gamma_vector_cat[m]
    gamma_avg = gamma_avg / len(w)
    #####################################################################################################################

    """# Received signal 구하기: 식(22)"""
    re_signal = torch.zeros(1, 2 * T).to(device)

    for m in range(len(w)):
        re_signal += gamma_vector_cat[m] * compressed_gradient[m] * sign_matrix[m]

    # 가우시안 노이즈 생성 및 노이즈 더하기
    noise = ((torch.randn(1, 2 * T).to(device)) / np.sqrt(2))
    re_signal_noise = re_signal+noise

    # 인원수로 나누고, 감마평균값으로 나누기
    re_signal_noise = re_signal_noise / sum_sign_matrix
    G_avg = re_signal_noise / (gamma_avg)


    """######################### 채널 환경 종료 #######################################"""

    # 패턴에 의해 값 원래 위치로 돌리기
    restored_gradient[Sp]=G_avg


    Fw_avg = copy.deepcopy(D)
    Fw_avg -= 0.01 * restored_gradient

    W_avg = recover_flattened(Fw_avg, L, model)
    w_avg = copy.deepcopy(w[0])

    j = 0
    for k in w_avg.keys():
        w_avg[k] = W_avg[j]
        j += 1

    return w_avg, g_ec


""""###########################################################################################"""
""""###########################################################################################"""

""""###########################################################################################"""
""""###########################################################################################"""


def FedAnalogPSGuide(w, previous_w_global, T, P, g_ec,po,device, K, add_error):
    ##############################################Add a devcie to 0th device

    model = []

    for k in w[0].keys():
        model.append(w[0][k].shape)

    W = []
    for k in w[0].keys():
        W.append(w[0][k])
    D, L = flatten_params(W)
    d = len(D)
    ##############################################################
    G = torch.zeros((len(w), d)).to(device)
    OG = torch.zeros((len(w), d)).to(device)
    restored_gradient = torch.zeros(d).to(device)
    #Sum = np.zeros(len(w))
    ##############################################################

    "새로 업데이트된 웨이트 값 플래튼 시키기"
    Fw = torch.zeros((len(w), d)).to(device)

    for i in range(len(w)):
        W = []
        for k in w[0].keys():
            W.append(w[i][k])

        D, L = flatten_params(W)

        D=D.reshape(21840)
        Fw[i]=D

        #for k in range(0, d):
        #    Fw[i][k] = D[k]

    "직전의 글로벌 웨이트 값 플래튼 시키기"
    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])
    D, L = flatten_params(W)

    "그래디언트 구하기"
    D=D.reshape(21840)
    for i in range(len(w)):
        G[i] = (D - Fw[i])/0.01 + g_ec[i]
        OG[i] = G[i]


    ###############################################################
    "Guide device sparsification"
    # 절대값을 내림차순으로 정렬
    Gprime = torch.sort(torch.abs(OG[0]),descending=True)[0]
    # 2T번째 값 구하기
    topk=Gprime[2*T-1]

    # topk보다 작은 값은 0으로 만들기
    indexing=torch.abs(OG[0])<topk
    # 1,0으로 표현하기
    SP = torch.ones(d).to(device)
    SP[indexing]=0

    # SP의 포지션 인덱스 값
    Sp=torch.nonzero(SP).reshape(2*T)


    #################################################################
    "유저별 패턴위치 값 추출하고 error 보존 "

    compressed_gradient = torch.zeros(len(w), 2 * T).to(device)
    pattern_gradient = torch.zeros(len(w), d).to(device)
    for i in range(1,len(w)):
        # 2T개만 추출하기
        compressed_gradient[i] = OG[i][Sp]
        # 에러보존하기
        pattern_gradient[i] = OG[i] * SP


        # 에러보존 여부
        if add_error:
            g_ec[i] = OG[i] - pattern_gradient[i]

    """#########################채널 환경 구성하기#######################################"""

    h2_matrix = {}  # 50 by 100 매트릭스
    s_matrix = {}  # 50 by 100 매트릭스
    sign_matrix = {}  # 100 by 100 매트릭스

    sampling = torch.distributions.exponential.Exponential(1.0)

    for m in range(1,len(w)):
        h2_matrix[m] = sampling.sample([1, T])

        # h^2 > 0.001 적용, 조건보다 작은값은 0으로 만들기
        s_matrix[m] = h2_matrix[m] > 0.001
        vector = np.vectorize(np.int)  # 브로드캐스팅을 위한 함수 설정
        s_matrix[m] = vector(s_matrix[m])  # 1,0으로 표시

        s_matrix[m] = torch.tensor(s_matrix[m]).to(device)  # 텐서 변환

        # s_matrix 복제해서 가로로 연결하기: 같은 페이딩 겪게 하기
        sign_matrix[m] = torch.cat([s_matrix[m], s_matrix[m]], axis=1)

        # sign_matrix의 합 구하기: 차후 나눠주기 위해서
    sum_sign_matrix = torch.zeros(1, 2 * T).to(device)
    for m in range(1,len(w)):
        sum_sign_matrix += sign_matrix[m]

    gradient_sum_timeslot = {}
    for m in range(1,len(w)):
        gradient_sum_timeslot[m] = sum(((((compressed_gradient[m] ** 2).reshape(2, T)).sum(axis=0)).reshape(int(T / 50), 50)).sum(axis=1))

    # E_1 구하기
    E_1 = 6.33153936413615
    # 4.726095458584443


    ####################################################################################################################
    # gamma 구하기
    gamma = []
    for m in range(1,len(w)):
        for i in range(int(T / 50)):
            gamma.append(torch.sqrt(T/50*po / gradient_sum_timeslot[m]) / np.sqrt(6.33153936413615))


    gamma_matrix = torch.tensor(gamma).reshape(len(w)-1, int(T / 50)).to(device)

    # timeslot 구간 별 gamma 평균 구하기

    gamma_vector_cat = {}
    for m in range(0,len(w)-1):
        gamma_vector_T = torch.zeros([int(T / 50), 50]).to(device)
        for r in range(int(T / 50)):
            # 각행에 같은 gamma 값 넣기,
            gamma_vector_T[r] = gamma_matrix[m][r]
        # 같은 감마값 (50),(50) ...(50).  50개 세트가 100개있음
        # 1*T 사이즈로 변경후 가로로 연결
        gamma_vector = gamma_vector_T.reshape(1, T)
        gamma_vector_cat[m] = torch.cat([gamma_vector, gamma_vector], axis=1)

    """# gamma 평균구하기: 식 (21)"""
    gamma_avg = torch.zeros(1, 2 * T).to(device)
    for m in range(0,len(w)-1):
        gamma_avg += gamma_vector_cat[m]
    gamma_avg = gamma_avg / (len(w)-1)
    #####################################################################################################################

    """# Received signal 구하기: 식(22)"""
    re_signal = torch.zeros(1, 2 * T).to(device)

    for m in range(1,len(w)):
        re_signal += gamma_vector_cat[m-1] * compressed_gradient[m] * sign_matrix[m]

    # 가우시안 노이즈 생성 및 노이즈 더하기
    noise = ((torch.randn(1, 2 * T).to(device)) / np.sqrt(2))
    re_signal_noise = re_signal + noise

    # 인원수로 나누고, 감마평균값으로 나누기
    re_signal_noise = re_signal_noise / sum_sign_matrix
    G_avg = re_signal_noise / (gamma_avg)


    """######################### 채널 환경 종료 #######################################"""


    # 패턴에 의해 값 원래 위치로 돌리기
    restored_gradient[Sp]=G_avg

    Fw_avg = copy.deepcopy(D)
    Fw_avg -= 0.01 * restored_gradient

    W_avg = recover_flattened(Fw_avg, L, model)
    w_avg = copy.deepcopy(w[0])

    j = 0
    for k in w_avg.keys():
        w_avg[k] = W_avg[j]
        j += 1

    return w_avg, g_ec








""""###########################################################################################"""
""""###########################################################################################"""

""""###########################################################################################"""
""""###########################################################################################"""
def FedAnalogAMP(w, previous_w_global,T,P,g_ec,po,add_error):


    # 모델의 shape 저장하기 "conv1.weight", "conv1.bias" ... ... ... 등
    #[torch.sizie([10,1,5,5]), torch.size([10]), torch.size([20,10,5,5]) ... ]
    model=[]

    for k in w[0].keys():
        model.append(w[0][k].shape)

    # W에 모든 파라미터 값을 리스트로 저장한다.

    W=[]
    for k in w[0].keys():
        W.append(w[0][k])

    # 플래튼시키기
    D,L=flatten_params(W)
    d=len(D)


    ##############################################################
    G=np.zeros((len(w),d))
    OG=np.zeros((len(w),d))
    Gprime=np.zeros(len(w))        
    Sum=np.zeros(len(w))
    rho=2*T/2
    ##############################################################
    Fw=np.zeros((len(w),d))


    """새로 업데이트된 웨이트 값 플래튼 시키기"""
    for i in range(0,len(w)):
        W=[]
        for k in w[0].keys():
            W.append(w[i][k])
        D,L=flatten_params(W)

        aa=D.to(device='cpu').reshape(21840)
        Fw[i]=aa
        # for k in range(0,d):
        #     Fw[i][k]=D[k]



    """이전 글로벌 웨이트 값 플래튼 시키기"""
    W=[]
    for k in w[0].keys():
        W.append(previous_w_global[k])

    #D,L=flatten_params(np.array(W))
    D, L = flatten_params(W)


    """ Gradient 구하기"""
    aa=D.to(device='cpu').reshape(21840)
    for i in range(0,len(w)):

        G[i]=(aa-Fw[i]) /0.01 +g_ec[i]



        OG[i]=G[i]


    
    
    ###############################################################
    
    zmean1=0
    zvar1=0


    """Sparsification"""
    for i in range(0, len(w)):
        Gprime[i]=np.sort(np.abs(G[i][:]))[::-1][int(rho)-1]  # int(rho) = 2*T/2= 5000
        # gradient를 절대값으로 취한 뒤, 큰값부더 작은갑 순으로 내림 차순으로 정리하고
        # 5000번째 값을 저장한다. (4999번)


    # 5000번째 값(중위값) 보다 작은 그래디언트는 0으로 만든다.

    for i in range(0, len(w)):
        small=np.abs(G[i])<Gprime[i]
        G[i][small]=0


        #에러보존  여부
        if add_error:
            g_ec[i]=OG[i]-G[i]

        zmean1+=G[i]


    zmean1=zmean1.sum()

        # for k in range(0, d):
        #     if np.abs(G[i][k])<Gprime[i]:
        #         G[i][k] = 0
        #     g_ec[i][k]=OG[i][k]-G[i][k]
        #
        #     #zmean1: 모든 유저의 그래디언트 값을 다 더하기 (0으로 변경된것 포함)
        #     zmean1+=G[i][k]


    # 모든 유저의 그래디언트 더한 값/ 5000 = 각 유저별 그래디언트 평균값 들의 합
    zmean1=zmean1/rho/len(w)
    #zmean1=zmean1/rho


    # SS: 모든 유저의 그래디언트 값 더하기
    # SSS: 모든 유저의 그래디언트 제곱값 더하기
    for i in range(0, len(w)):

        SS=G[i].sum()
        SSS=(G[i]**2).sum()

        # SS=0
        # SSS=0
        # for k in range(0, d):
        #     SS+=G[i][k]
        #     SSS+=G[i][k]**2


        # 분산 합
        zvar1 += (SSS/rho/(len(w)**2)) - (SS / rho/len(w)) ** 2
        #zvar1+=SSS/rho-(SS/rho)**2
    
            

    ###################################################################################            

    """컴프레싱하기 """
    # 평균0, 분산: np.sqrt(d) 인 가우시안 분포를 가지는 (2T X d)사이즈의 임의 매트릭스 생성
    #A = np.random.normal(0,np.sqrt(d),(2*T, d))
    #A = np.random.normal(0, np.sqrt(2*T), (2 * T, d))
    A = np.random.normal(0, np.sqrt(1/(2 * T)), (2 * T, d))



    # A매트릭스와 곱하여서 압축하기  d개(21840) -> 2T개(10000개)
    Gcomp=np.zeros((len(w),2*T))
    for i in range(0, len(w)):
        Gcomp[i]=np.dot(A,G[i])
        # Gcomp[i]: (2T X d) x (d X 1) = (2T X 1) 사이즈



    ########################################################################################

    # 이전방식

    # Gcomp_avg=np.zeros(2*T)
    # for i in range(0, len(w)):
    #     for k in range(0, 2*T):
    #         Gcomp_avg[k]+=Gcomp[i][k]
    #         # 모든 유저의 압축된 그래디언트 값 더하기
    #         Sum[i]+=Gcomp[i][k]**2
    #         # 각 유저별 그래디언트 제곱값 더하기



    """#########################채널 환경 구성하기#######################################"""



    # Gcomp -> 100 by 100 matrix로 변경
    # Gcomp_matrix={}
    # for m in range(len(w)):
    #     Gcomp_matrix[m]=Gcomp[m].reshape(100,100)



    h2_matrix={}     #  50 by 100 매트릭스
    s_matrix = {}   # 50 by 100 매트릭스
    sign_matrix = {} # 100 by 100 매트릭스

    sampling=torch.distributions.exponential.Exponential(1.0)

    for m in range(len(w)):

        h2_matrix[m]=sampling.sample([1,T])
        # h^2의 매트릭스 구하기 (50 x 100) : real값과 imag 값 둘다 같은 페이딩 받기때문에, rows=50으로 설정
        # h_matrix[m]=np.zeros((50,100))
        # h_matrix[m] += np.random.exponential(1, (50,100))


        # h^2 > 0.001 적용, 조건보다 작은값은 0으로 만들기
        s_matrix[m]=h2_matrix[m]>0.001
        vector=np.vectorize(np.int) # 브로드캐스팅을 위한 함수 설정
        s_matrix[m]=vector(s_matrix[m]) # 1,0으로 표시

        # s_matrix 복제해서 가로로 연결하기: 같은 페이딩 겪게 하기
        sign_matrix[m]=np.concatenate([s_matrix[m],s_matrix[m]], axis=1)


    # sign_matrix의 합 구하기: 차후 나눠주기 위해서
    sum_sign_matrix=np.zeros([1,2*T])
    for m in range(len(w)):
        sum_sign_matrix += sign_matrix[m]


    # # gradient l2 구학
    # gradient_sum_timeslot={}
    # for m in range(len(w)):
    #     gradient_sum_timeslot[m]=(Gcomp[m]**2).sum()

    """Power allocation: 찬호논문 버전"""
    gradient_sum_timeslot={}

    # for m in range(len(w)):
    #     gradient_sum_timeslot[m]=((((Gcomp[m]**2).reshape(2,T)).sum(axis=0)).reshape(int(T/50),50)).sum(axis=1)


    """Power allocation: 진현논문 버전"""
    for m in range(len(w)):
        gradient_sum_timeslot[m] = sum(((((Gcomp[m]**2).reshape(2, T)).sum(axis=0)).reshape(int(T / 50), 50)).sum(axis=1))


    # E_1 구하기
    E_1=6.33153936413615
    #4.726095458584443

    ################################################ N=1고정방식  ######################################################
    # """"# gamma 구하기 식(20)"""
    # gamma=[]
    # for m in range(len(w)):
    #     gamma.append((np.sqrt(np.random.exponential(1,(1,1)))) * (np.sqrt(po / gradient_sum_timeslot[m]) / np.sqrt(E_1)))
    #
    #
    # """# gamma 평균구하기: 식 (21)"""
    # gamma_avg=torch.zeros(1)
    # for m in range(len(w)):
    #     gamma_avg += gamma[m]
    # gamma_avg=gamma_avg/len(w)
    ##################################################################################################################


    ####################################################################################################################
    """"# gamma 구하기 식(20): 찬호버전"""
    gamma=[]
    # for m in range(len(w)):
    #     for i in range(int(T/50)):
    #         gamma.append(((np.sqrt(po/gradient_sum_timeslot[m][i])/np.sqrt(E_1))))

    """"# gamma 구하기 식(20): 진현버전"""
    for m in range(len(w)):
        for i in range(int(T / 50)):
            gamma.append(np.sqrt(T / 50 * po / gradient_sum_timeslot[m]) / np.sqrt(6.33153936413615))

    gamma_matrix=np.array(gamma).reshape(len(w), int(T/50))


    # timeslot 구간 별 gamma 평균 구하기
    #gamma_avg=(gamma_t_array.sum(axis=0))/len(w)

    gamma_vec_cat={}
    for m in range(len(w)):
        gamma_vec_T=np.zeros([int(T/50),50])
        for r in range(int(T/50)):

            gamma_vec_T[r]=gamma_matrix[m][r]
        gamma_vec=gamma_vec_T.reshape(1,T)
        gamma_vec_cat[m]=np.concatenate([gamma_vec,gamma_vec], axis=1)


    # gamma 평균구하기
    gamma_avg=np.zeros([1,2*T])
    for m in range(len(w)):
        gamma_avg+=gamma_vec_cat[m]
    gamma_avg=gamma_avg/len(w)
    #####################################################################################################################

    # Received signal 구하기
    received_signal=np.zeros([1,2*T])

    for m in range(len(w)):

        received_signal += gamma_vec_cat[m]*Gcomp[m]*sign_matrix[m]
        #received_signal += (gamma_t_array[m].reshape(1,100))*(Gcomp_matrix[m])*(sign_matrix[m])

    # 노이즈 더하기
    received_signal +=((np.random.normal(0,1,(1,2*T)))/np.sqrt(2))



    # 인원수로 나누고, 감마평균값으로 나누기
    received_signal=received_signal/sum_sign_matrix
    Gcomp_avg=received_signal/(gamma_avg)






    # """이전 방식: 동일 페이딩"""
    # # 감마값 구하기
    # gamma=np.sqrt(np.random.exponential(1,(1,1)))*np.sqrt(P/Sum[0])
    #
    # # 최소 감마값 구하기
    # for i in range(1, len(w)):
    #     gamma=min(gamma,np.sqrt(np.random.exponential(1,(1,1)))*np.sqrt(P/Sum[i]))
    #
    # # 노이즈 값 더하기
    # for k in range(0, 2*T):
    #     Gcomp_avg[k]+=np.random.normal(0,1,(1,1))/gamma/np.sqrt(2)
    #
    # ipdb.set_trace()



    ##########################################################################################
    #AMP apply   & divide by len(w)  Gcomp_avg -->  recovery --> divide by len(w) -->G_avg
    
    #start=time.time()

    ny=2*T
    nz=d
    wvar=0
    #wvar = 1 / 2 / (gamma_avg_all ** 2) # 노이즈 어떻게 될지..
    #wvar=1/2/(gamma_**2)
    zshape = (nz,)   # Shape of z matrix
    yshape = (ny,)   # Shape of y matrix
    Ashape = (ny,nz)   # Shape of A matrix


    est0 = vp.estim.DiscreteEst(0,1,zshape)
    est1 = vp.estim.GaussEst(zmean1,zvar1,zshape)
    


    est_list = [est0, est1]
    prob_on=rho/d
    pz = np.array([1-prob_on, prob_on])
    est_in = vp.estim.MixEst(est_list, w=pz, name='Input')
    Aop = vp.trans.MatrixLT(A,zshape)
    est_out = vp.estim.LinEst(Aop,Gcomp_avg[0],wvar,map_est=False, name='Output')
    msg_hdl = vp.estim.MsgHdlSimp(map_est=False, shape=zshape)

    
    nit = 20  # number of iterations
    solver = vp.solver.Vamp(est_in,est_out,msg_hdl,hist_list=['zhat', 'zhatvar'],nit=nit)
    solver.solve()
    G_avg=solver.zhat

    #G_avg=G_avg/len(w) 복원전에 나눴기 때문에 지웠음....






    ##########################################################################################


    "웨이트 업데이트"
    #Fw_avg=copy.deepcopy(D.to(device='cpu'))
    Fw_avg = copy.deepcopy(aa) # D의 Cpu할당버전= aa

    Fw_avg-=0.01*G_avg

    # for k in range(0, d):
    #     Fw_avg[k]-=0.01*G_avg[k]
    #     # 모멘텀 주기
    #     # Fw_avg[k] = 0.9*Fw_avg[k] - 0.01*G_avg[k]




    W_avg = recover_flattened(Fw_avg,L,model)
    w_avg=copy.deepcopy(w[0])
    
    j=0
    for k in w_avg.keys():
        w_avg[k]=W_avg[j]
        j+=1

    
    return w_avg, g_ec



""""###########################################################################################"""
""""###########################################################################################"""

""""###########################################################################################"""
""""###########################################################################################"""


def Fed_DDSGD(w, previous_w_global,T,P,g_ec,po, device):

    """############"""
    """채널 관련 처리"""
    # 개인별 리소스 할당
    resource= int(T/len(w)) # T=5000 기준, m=50 기준, 개별 T=100
    # 서브캐리어 10으로 고정
    subcarrier=10
    #subcarrier=17
    #timeslot=1
    # 개인별 타임슬롯 구하기
    timeslot= int(resource/subcarrier) # T=100, SUB=10, TIMESLOT=10
    device_power= po*(T/50)


    m=torch.distributions.exponential.Exponential(1.0)
    h2_matrix=m.sample([timeslot*len(w),subcarrier]).to(device) # T5000기준, (10*50)*10 매트릭스

    # 각 유저에게 파워 할당하기
    tol = 1e-2
    p_constrain = device_power/timeslot # po= 1, T=5000일때. 한사람당 파워 100할당, 한사람당 타임슬롯별 10 // po30, 한사람당 파워 900, 타임슬롯별 150
    number_vector=h2_matrix[0].shape[0]
    all_p_allocation=torch.zeros([timeslot*len(w),subcarrier]).to(device)


    for i in range(int(timeslot*len(w))):
        noise_vec=1/h2_matrix[i]
        water_line=min(noise_vec)+p_constrain/number_vector # 초기 워터라인 설정
        p_allocation = water_line - noise_vec

        for r in range(number_vector):  # 파워 값 마이너스 일 경우 0으로 설정
            if p_allocation[r] < 0:
                p_allocation[r] = 0

        p_tot = sum(p_allocation)  # power total 구하기


        while abs(p_constrain - p_tot) > tol:  # 할당된 파워가 파워 컨스트레인 보다 작으면 반복 수행
            water_line = water_line + (p_constrain - p_tot) / number_vector
            p_allocation = water_line - noise_vec
            for j in range(number_vector):
                if p_allocation[j] < 0:
                    p_allocation[j] = 0
            p_tot = sum(p_allocation)


        all_p_allocation[i]+=p_allocation



    # 각 유저와 중앙서버 간의 capacity 구하기
    p_allo_h2_matrix=all_p_allocation*h2_matrix
    p_allo_h2_matrix=p_allo_h2_matrix.reshape(len(w),-1) # 사람수로 매트릭스 조정
    p_allo_h2_matrix+=1
    capacitylog=torch.log2(p_allo_h2_matrix)
    capacity=capacitylog.sum(axis=1)



    """############"""
    """ 파라미터 처리"""
    # 모델 shape 저장
    model = []
    for k in w[0].keys():
        model.append(w[0][k].shape)

    # W에 모든 파라미터 값을 리스트로 저장한다.
    W = []
    for k in w[0].keys():
        W.append(w[0][k])

    D, L = flatten_params(W)
    # D, L = flatten_params(np.array(W))
    # D= 플래튼 값
    # L= 인덱스

    d = len(D)

    ##############################################################
    G = torch.zeros((len(w), d)).to(device)  # [유저 * 플래튼 값]
    OG = torch.zeros((len(w), d)).to(device)
    # Gprime = np.zeros(len(w))
    # Sum = np.zeros(len(w))
    # rho = 2 * T / 2
    ##############################################################

    """새로 업데이트된 웨이트 값 플래튼 시키기"""
    Fw = torch.zeros((len(w), d)).to(device)

    for i in range(0, len(w)):
        W = []
        for k in w[0].keys():
            W.append(w[i][k])
        D, L = flatten_params(W)
        # D, L = flatten_params(np.array(W))

        D=D.reshape(21840)
        Fw[i]=D

        # Fw[0]= [ d개의 플래튼 된 값이 들어있음.  ]
        # Fw[1]= [ d개의 플래튼 된 값이 들어있음.  ]
        # Fw[2]= [ d개의 플래튼 된 값이 들어있음.  ]
        #  ... ... ...
        # Fw[9]= [ d개의 플래튼 된 값이 들어있음.  ]

        # for문 연산 느려서 바꿈.
        # for k in range(0, d):
        #     Fw[i][k] = D[k]

    """ 직전의 글로벌 웨이트 값 플래튼 시키기"""
    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])
    D, L = flatten_params((W))


    """ Gradient 구하기"""
    D = D.reshape(21840)
    for i in range(len(w)):
        G[i]=(D-Fw[i]) / 0.01 +g_ec[i]
        OG[i]=G[i]

    """ Gradient 구하기""" # 넘나 느린방식
    # for i in range(0, len(w)):
    #     for k in range(0, d):
    #         G[i][k] = (D[k] - Fw[i][k]) / 0.01 +g_ec[i][k]  # {그래디언트 =( 직전웨이트-새로운웨이트) / lr} + error accumulation
    #         OG[i][k] = G[i][k]  # OG는 gradient값




    """############"""
    """q_t 구하기"""
    # Combination 함수 생성
    def nCr(n, r):
        if n < 1 or r < 0 or n < r:
            raise ValueError
        r = min(r, n - r)
        numerator = reduce(op.mul, range(n, n - r, -1, ), 1)
        denominator = reduce(op.mul, range(1, r + 1), 1)
        return numerator // denominator


    # # Upperbound 구하기
    # upper=torch.zeros(len(w))
    # for i in range(len(w)):
    #     upper[i]=2**(int(capacity[i])-33)



    # q_t 개수 구하기
    all_number_qt=torch.zeros(len(w)).to(device)
    for i in range(len(w)):
        number_qt = 1
        while nCr(d, number_qt) < 2**(int(capacity[i])-33):
            number_qt += 1
        # 최종 q_t 개수: while문에서 마지막에 1 더해진 값 뺴기
        number_qt -= 1
        #채널 에러 방지
        if number_qt==0:
            number_qt=1

        all_number_qt[i]=number_qt



    # 유저별 q_t 구하기
    qt_gradient = torch.zeros([len(w),21840]).to(device)
    index_info=torch.zeros([21840]).to(device)


    # qt 개수가지고 gradient 처리하기
    for i in range(len(w)):
        positive_qt, index_po = torch.topk(OG[i], int(all_number_qt[i]), largest=True)
        negative_qt, index_ne = torch.topk(OG[i], int(all_number_qt[i]), largest=False)
        positive_mean = torch.mean(positive_qt)
        negative_mean = abs(torch.mean(negative_qt))

        if positive_mean>negative_mean:
            qt_gradient[i][index_po]=positive_mean
            #index_info[index_po] += 1
        else:
            qt_gradient[i][index_ne] = torch.mean(negative_qt)
            #index_info[index_ne]+=1

        # error 누적하기
        g_ec[i]=OG[i]-qt_gradient[i]


    # 0인 값에 대해서는 1을 더해주기 (나눠주기 위해서)
    #index_info[torch.nonzero(index_info==0)]+=1


    # q_t avg값 구하기
    qt_avg=(qt_gradient.sum(axis=0))/len(w)





    """weight 업데이트"""
    Fw_avg = copy.deepcopy(D)
    Fw_avg -=0.01 * qt_avg

    # 넘나 느린방식.
    # for k in range(0, d):
    #     Fw_avg[k] -= 0.01 * qt_avg[k]


    """ flatten -> 매트릭스"""
    W_avg = recover_flattened(Fw_avg, L, model)

    """ 모델 매트릭스에 값 넣기"""
    w_avg = copy.deepcopy(w[0])
    j = 0
    for k in w_avg.keys():
        w_avg[k] = W_avg[j]
        j += 1



    return w_avg, g_ec


""""###########################################################################################"""
""""###########################################################################################"""

""""###########################################################################################"""
""""###########################################################################################"""



def sign_sgd(w, previous_w_global,T,P,g_ec,po, device):

    """############"""
    """채널 관련 처리"""
    # h^2 샘플링
    # 한사람당 리소스 할당량
    resource=int(T/len(w)) # T=5000, m=50, T=100 // T=1000, m=50, T=20
    # 서브캐리어 10 설정,
    subcarrier=10
    # 타임슬롯 구하기
    timeslot= int(resource/subcarrier) #T=20, TIMESOT=2

    m=torch.distributions.exponential.Exponential(1.0)
    h2_matrix=m.sample([timeslot*len(w),subcarrier]).to(device) # T5000기준, 100*50 매트릭스


    # 각 유저에게 파워 할당하기
    tol = 1e-2
    p_constrain = po*100/timeslot # po 50일때. 한사람당 파워 5000할당, 한사람당 타임슬롯별 2500 // po30, 한사람당 파워 3000, 타임슬롯별 1500

    number_vector=h2_matrix[0].shape[0]
    all_p_allocation=torch.zeros([timeslot*len(w),subcarrier]).to(device)



    for i in range(int(T/subcarrier)):
        noise_vec=1/h2_matrix[i]
        water_line=min(noise_vec)+p_constrain/number_vector # 초기 워터라인 설정
        p_allocation = water_line - noise_vec

        for r in range(number_vector):  # 파워 값 마이너스 일 경우 0으로 설정
            if p_allocation[r] < 0:
                p_allocation[r] = 0

        p_tot = sum(p_allocation)  # power total 구하기

        while abs(p_constrain - p_tot) > tol:  # 할당된 파워가 파워 컨스트레인 보다 작으면 반복 수행
            water_line = water_line + (p_constrain - p_tot) / number_vector
            p_allocation = water_line - noise_vec
            for j in range(number_vector):
                if p_allocation[j] < 0:
                    p_allocation[j] = 0
            p_tot = sum(p_allocation)

        all_p_allocation[i]+=p_allocation

    # 각 유저와 중앙서버 간의 capacity 구하기
    p_allo_h2_matrix=all_p_allocation*h2_matrix
    p_allo_h2_matrix=p_allo_h2_matrix.reshape(len(w),-1) # 사람수로 매트릭스 조정
    p_allo_h2_matrix+=1
    capacitylog=torch.log2(p_allo_h2_matrix)
    capacity=capacitylog.sum(axis=1)



    """############"""
    """ 파라미터 처리"""
    # 모델 shape 저장
    model = []
    for k in w[0].keys():
        model.append(w[0][k].shape)

    # W에 모든 파라미터 값을 리스트로 저장한다.
    W = []
    for k in w[0].keys():
        W.append(w[0][k])

    D, L = flatten_params(W)
    # D, L = flatten_params(np.array(W))
    # D= 플래튼 값
    # L= 인덱스

    d = len(D)

    ##############################################################
    G = torch.zeros((len(w), d)).to(device)  # [유저 * 플래튼 값]
    OG = torch.zeros((len(w), d)).to(device)
    ##############################################################

    """새로 업데이트된 웨이트 값 플래튼 시키기"""
    Fw = torch.zeros((len(w), d)).to(device)

    for i in range(0, len(w)):
        W = []
        for k in w[0].keys():
            W.append(w[i][k])
        D, L = flatten_params(W)
        D=D.reshape(21840)
        Fw[i]=D

    """ 직전의 글로벌 웨이트 값 플래튼 시키기"""
    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])
    D, L = flatten_params((W))

    """ Gradient 구하기"""
    D = D.reshape(21840)
    for i in range(len(w)):
        G[i]=(D-Fw[i]) / 0.01 +g_ec[i]
        OG[i]=G[i]

    """############"""
    """q_t 구하기"""
    # Combination 함수 생성
    def nCr(n, r):
        if n < 1 or r < 0 or n < r:
            raise ValueError
        r = min(r, n - r)
        numerator = reduce(op.mul, range(n, n - r, -1, ), 1)
        denominator = reduce(op.mul, range(1, r + 1), 1)
        return numerator // denominator

    all_number_qt=torch.zeros(len(w)).to(device)
    for i in range(len(w)):
        number_qt = 1
        while ((nCr(d, number_qt)) - (2 ** (int(capacity[i]) - number_qt))) < 0:  # capacity를 int형으로 변환해도 되는지? infinity 막기 위해서: 결론 별차이 없다.
            number_qt += 1
        # 최종 q_t
        number_qt -= 1
        all_number_qt[i]=number_qt



    #유저별 qt 구하기
    qt_gradient = torch.zeros([len(w),21840]).to(device)
    index_info=torch.zeros([21840]).to(device)

    # TOPK 값 받기

    for i in range(len(w)):
        q_t, index_qt = torch.topk(abs(OG[i]), int(all_number_qt[i]), largest=True)

        sign_G = torch.zeros(int(all_number_qt[i])).to(device)

        # 0보다 크면 1, 0보다 작으면 -1로 설정하기
        for j in range(int(all_number_qt[i])):
            if OG[i][index_qt][j]>0:
                sign_G[j]=1
            else:
                sign_G[j]=-1

        #index_info[index_qt]+=1
        #인덱스에 sign_G값 넣고, 나머지 ZERO화
        qt_gradient[i][index_qt]=sign_G

        # error 누적하기
        OG[i][index_qt]=0
        g_ec[i]=OG[i]
        #g_ec[i] = OG[i] - qt_gradient[i]


    #index_info[torch.nonzero(index_info==0)]+=1

    #qt_avg=(qt_gradient.sum(axis=0))/index_info
    qt_avg = (qt_gradient.sum(axis=0)) # len(w) 안나눠져야 성능이 나온다..


    """weight 업데이트"""
    Fw_avg = copy.deepcopy(D)
    Fw_avg -=0.01 * qt_avg

    """ flatten -> 매트릭스"""
    W_avg = recover_flattened(Fw_avg, L, model)

    """ 모델 매트릭스에 값 넣기"""
    w_avg = copy.deepcopy(w[0])
    j = 0
    for k in w_avg.keys():
        w_avg[k] = W_avg[j]
        j += 1

    return w_avg, g_ec


""""###########################################################################################"""
""""###########################################################################################"""
""""###########################################################################################"""
""""###########################################################################################"""
""""###########################################################################################"""
""""###########################################################################################"""
def sign_major(w, previous_w_global,T,P,g_ec,po, device):


    """############"""
    """채널 관련 처리"""
    # h^2 샘플링
    # 한사람당 리소스 할당량
    resource=int(T/len(w)) # T=5000, m=50, T=100 // T=1000, m=50, T=20
    # 서브캐리어 10 설정,
    subcarrier=10
    # 타임슬롯 구하기
    timeslot= int(resource/subcarrier) #T=100, TIMESOT=10
    #subcarrier=17
    #timeslot=1
    device_power=po*(T/50)

    m=torch.distributions.exponential.Exponential(1.0)
    h2_matrix=m.sample([timeslot*len(w),subcarrier]).to(device) # T5000기준, 100*50 매트릭스



    # 각 유저에게 파워 할당하기
    tol = 1e-2
    p_constrain = device_power/timeslot

    number_vector=h2_matrix[0].shape[0]
    all_p_allocation=torch.zeros([timeslot*len(w),subcarrier]).to(device)

    for i in range(timeslot*len(w)):
        noise_vec=1/h2_matrix[i]
        water_line=min(noise_vec)+p_constrain/number_vector # 초기 워터라인 설정
        p_allocation = water_line - noise_vec

        for r in range(number_vector):  # 파워 값 마이너스 일 경우 0으로 설정
            if p_allocation[r] < 0:
                p_allocation[r] = 0

        p_tot = sum(p_allocation)  # power total 구하기

        while abs(p_constrain - p_tot) > tol:  # 할당된 파워가 파워 컨스트레인 보다 작으면 반복 수행
            water_line = water_line + (p_constrain - p_tot) / number_vector
            p_allocation = water_line - noise_vec
            for j in range(number_vector):
                if p_allocation[j] < 0:
                    p_allocation[j] = 0
            p_tot = sum(p_allocation)

        all_p_allocation[i]+=p_allocation


    # 각 유저와 중앙서버 간의 capacity 구하기
    p_allo_h2_matrix=all_p_allocation*h2_matrix
    p_allo_h2_matrix+=1
    # 유저별로 나누기
    p_allo_h2_matrix=p_allo_h2_matrix.reshape(len(w),-1)

    capacitylog=torch.log2(p_allo_h2_matrix)
    capacity=capacitylog.sum(axis=1)


    """############"""
    """ 파라미터 처리"""
    # 모델 shape 저장
    model = []
    for k in w[0].keys():
        model.append(w[0][k].shape)

    # W에 모든 파라미터 값을 리스트로 저장한다.
    W = []
    for k in w[0].keys():
        W.append(w[0][k])

    D, L = flatten_params(W)
    # D, L = flatten_params(np.array(W))
    # D= 플래튼 값
    # L= 인덱스

    d = len(D)

    ##############################################################
    G = torch.zeros((len(w), d)).to(device)  # [유저 * 플래튼 값]
    OG = torch.zeros((len(w), d)).to(device)
    ##############################################################

    """새로 업데이트된 웨이트 값 플래튼 시키기"""
    Fw = torch.zeros((len(w), d)).to(device)

    for i in range(0, len(w)):
        W = []
        for k in w[0].keys():
            W.append(w[i][k])
        D, L = flatten_params(W)
        D=D.reshape(21840)
        Fw[i]=D

    """ 직전의 글로벌 웨이트 값 플래튼 시키기"""
    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])
    D, L = flatten_params((W))

    """ Gradient 구하기"""
    D = D.reshape(21840)
    for i in range(len(w)):
        G[i]=(D-Fw[i]) / 0.01 +g_ec[i]
        OG[i]=G[i]

    """############"""
    """q_t 구하기"""
    # Combination 함수 생성
    def nCr(n, r):
        if n < 1 or r < 0 or n < r:
            raise ValueError
        r = min(r, n - r)
        numerator = reduce(op.mul, range(n, n - r, -1, ), 1)
        denominator = reduce(op.mul, range(1, r + 1), 1)
        return numerator // denominator



    all_number_qt=torch.zeros(len(w)).to(device)
    for i in range(len(w)):
        number_qt = 1
        while ((nCr(d, number_qt)) - (2 ** (int(capacity[i]) - number_qt))) < 0:  # capacity를 int형으로 변환해도 되는지? infinity 막기 위해서: 결론 별차이 없다.
            number_qt += 1
        # 최종 q_t
        number_qt -= 1
        all_number_qt[i]=number_qt


    #유저별 qt 구하기
    qt_gradient =torch.zeros([len(w), 21840]).to(device)
    index_info=torch.zeros([21840]).to(device)

    # TOPK 값 받기
    for i in range(len(w)):
        q_t, index_qt = torch.topk(abs(OG[i]), int(all_number_qt[i]), largest=True)

        sign_G = torch.zeros(int(all_number_qt[i])).to(device)

        # 0보다 크면 1, 0보다 작으면 -1로 설정하기
        for j in range(int(all_number_qt[i])):
            if OG[i][index_qt][j]>0:
                sign_G[j]=1
            else:
                sign_G[j]=-1

        #index_info[index_qt]+=1
        #인덱스에 sign_G값 넣고, 나머지 ZERO화
        qt_gradient[i][index_qt]=sign_G

        # error 누적하기 (에러 통째로 날리는 방식이 성능이 좋음)
        OG[i][index_qt]=0
        g_ec[i] = OG[i]
        # g_ec[i] = OG[i] - qt_gradient[i]

    #index_info[torch.nonzero(index_info==0)]+=1

    #qt_avg=(qt_gradient.sum(axis=0))/index_info
    #qt_avg = (qt_gradient.sum(axis=0))/len(w)

    # majority rule

    qt_sum= (qt_gradient.sum(axis=0))
    qt_sum_index=torch.nonzero(qt_sum)
    majority_qt=torch.zeros([21840]).to(device)
    for i in range(qt_sum_index.shape[0]):
        if qt_sum[qt_sum_index[i]] < 0:
            majority_qt[qt_sum_index[i]]= -1

        else:
            majority_qt[qt_sum_index[i]] = 1



    """weight 업데이트"""
    Fw_avg = copy.deepcopy(D)
    Fw_avg -=0.01 * majority_qt

    """ flatten -> 매트릭스"""
    W_avg = recover_flattened(Fw_avg, L, model)

    """ 모델 매트릭스에 값 넣기"""
    w_avg = copy.deepcopy(w[0])
    j = 0
    for k in w_avg.keys():
        w_avg[k] = W_avg[j]
        j += 1

    return w_avg, g_ec