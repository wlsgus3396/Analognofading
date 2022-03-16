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

from models import vampyre as vp
from csvec import CSVec

import operator as op
from functools import reduce

""""###########################################################################################"""
""""###########################################################################################"""


def FedAvg(w, device):
    model = []

    for k in w[0].keys():
        model.append(w[0][k].shape)

    # W에 w[0] 유저의 모든 파라미터 값을 리스트로 저장한다.
    W = []

    for k in w[0].keys():
        W.append(w[0][k])

    D, L = flatten_params(np.array(W))
    d = len(D)
    # D, L = flatten_params(np.array(W))
    # 리스트를 플래튼 시킨다.
    # D= 플래튼 값 21840개, tensor
    # L= 인덱스

    Fw = torch.zeros((len(w), d)).to(device)
    # shape: user수 * 21840

    for i in range(0, len(w)):
        W = []
        for k in w[0].keys():
            W.append(w[i][k])
        # W에  모든 파라미터를 저장

        D, L = flatten_params(np.array(W))

        D = D.reshape(21840)
        Fw[i] = D

        # for k in range(0,d):
        #     Fw[i][k]=D[k]
        #     #Fw에 각 유저의 파라미터를 순차적으로 저장

    Fw_avg = copy.deepcopy(D)

    Fw_avg[:] = 0
    Fw_avg = Fw.sum(axis=0)
    Fw_avg = Fw_avg / len(w)

    # for k in range(0, d):
    #     Fw_avg[k]=0
    #     # Fw_avg 값 초기화
    #     for i in range(0, len(w)):
    #         Fw_avg[k] += Fw[i][k]
    #     Fw_avg[k] = Fw_avg[k]/len(w)
    #     # 더해서 평균 구하기

    W_avg = recover_flattened(Fw_avg, L, model)
    # 파라미터 형태로 복원

    w_avg = copy.deepcopy(w[0])
    j = 0
    for k in w_avg.keys():
        w_avg[k] = W_avg[j]
        j += 1

    # w_avg에 넣기

    return w_avg


""""###########################################################################################"""
""""###########################################################################################"""

""""###########################################################################################"""
""""###########################################################################################"""


def FedSketch(w, previous_w_global, T, P, K, cc, g_ec, s_momentum, po, add_momentum, device):
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
        D = D.reshape(21840)
        Fw[i] = D

        # for k in range(0, d):
        #     Fw[i][k] = D[k]

        # Fw[0]= [ d개의 플래튼 된 값이 들어있음.  ]
        # Fw[1]= [ d개의 플래튼 된 값이 들어있음.  ]
        # Fw[2]= [ d개의 플래튼 된 값이 들어있음.  ]
        #  ... ... ...
        # Fw[9]= [ d개의 플래튼 된 값이 들어있음.  ]

    """ 직전의 글로벌 웨이트 값 플래튼 시키기"""
    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])
    D, L = flatten_params((W))

    """ Gradient 구하기"""
    D = D.reshape(21840)
    for i in range(len(w)):
        G[i] = (D - Fw[i]) / 0.01 + g_ec[i]
        OG[i] = G[i]

        # for k in range(0, d):
        #     G[i][k] = (D[k] - Fw[i][k]) / 0.01   # {그래디언트 =( 직전웨이트-새로운웨이트) / lr} + error accumulation
        #     OG[i][k] = G[i][k]  # OG는 gradient값

    """컴프레싱"""
    # Gcomp = np.zeros((len(w), 2 * T))

    dd = d
    ccrr = 2 * T  # 10000 / 2000
    rr = int(ccrr / cc)  # 25    /   10

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

    # """넘파이값 -> 텐서로 변환"""
    # Tensor_OG = []
    # for i in range(len(w)):
    #     Tensor_OG.append(OG[i])
    #     #Tensor_OG.append(torch.Tensor(OG[i]))
    #     # OG[0]= 21840개의 그래디언트 값

    Sketch = [Sketch_0, Sketch_1, Sketch_2, Sketch_3, Sketch_4, Sketch_5, Sketch_6, Sketch_7, Sketch_8, Sketch_9,
              Sketch_10, Sketch_11, Sketch_12, Sketch_13, Sketch_14, Sketch_15, Sketch_16, Sketch_17, Sketch_18,
              Sketch_19,
              Sketch_20, Sketch_21, Sketch_22, Sketch_23, Sketch_24, Sketch_25, Sketch_26, Sketch_27, Sketch_28,
              Sketch_29,
              Sketch_30, Sketch_31, Sketch_32, Sketch_33, Sketch_34, Sketch_35, Sketch_36, Sketch_37, Sketch_38,
              Sketch_39,
              Sketch_40, Sketch_41, Sketch_42, Sketch_43, Sketch_44, Sketch_45, Sketch_46, Sketch_47, Sketch_48,
              Sketch_49]

    """스케칭  x.table에 스케칭한 값이 들어있음."""
    for i in range(len(w)):
        # Sketch[i].accumulateVec(Tensor_OG[i])
        Sketch[i].accumulateVec(OG[i])
        # EX) Sketch[0].table.shape= 100 by 100

    """타임슬롯 및 ofdm 적용방식"""
    # sub carrier= 50개
    # Time slot= 100개 가정
    h2_matrix = {}  # 100 by 50 매트릭스
    s_matrix = {}  # 100 by 50 매트릭스
    sign_matrix = {}  # 100 by 100 매트릭스

    # exponential 샘플링
    sampling = torch.distributions.exponential.Exponential(1.0)

    for m in range(len(w)):
        # h^2의 매트릭스 구하기 1*T SIZE
        h2_matrix[m] = sampling.sample([1, T])

        # h^2 > 0.001 적용, 조건보다 작은값은 0으로 만들기
        s_matrix[m] = h2_matrix[m] > 0.001
        vector = np.vectorize(np.int)  # 브로드캐스팅을 위한 함수 설정
        s_matrix[m] = vector(s_matrix[m])  # 1,0으로 표시

        # s_matrix 복제해서 가로로 연결하기: 같은 페이딩 겪게 하기 (1*2T)
        sign_matrix[m] = np.concatenate([s_matrix[m], s_matrix[m]], axis=1)

    # 넘파이 -> 텐서
    for i in range(len(w)):
        sign_matrix[i] = torch.tensor(sign_matrix[i]).to(device)

    """# sign_matrix의 합 구하기: 차후 나눠주기 위해서..식(14) """

    sum_sign_matrix = torch.zeros(1, 2 * T).to(device)
    for m in range(len(w)):
        sum_sign_matrix += sign_matrix[m]

    """# E_1(X) 함수 만들기: 근사치로 사용"""
    x = Symbol('x')
    f = exp(-x) / x
    # E_1(0.001)로 하기
    E_1 = Integral(f, (x, 0.001, S.Infinity)).doit().evalf()
    # .doit() 실행 메소드
    # .evalf() 숫자로 나오게 하기

    """근사치"""
    # E_1 근사치: 0.001일때, 6.33153936413615
    # E_1 근사치: 0.005일때, 4.726095458584443

    #######################   TIME SLOT 1로 할 경우######################################
    # # gradient의 l2 값 구하기: 100 by 100 행렬에서 각 행을 같은 타임슬롯으로 가정
    #
    # gradient_sum_timeslot = {}
    # sketch_to_vector={}
    #
    # for m in range(len(w)):
    #     # sketchmatrix(rr,cc) -> vector(1,2T)
    #     sketch_to_vector[m]=((Sketch[m].table).reshape(1,2*T))
    #     # 제곱 -> (1,2T) -> (2,T) 만들어서 더하기 -> (1,T) -> (100,50) 변형 -> 가로로 더함. (100,1) 은 타임슬롯별 그래디언트 제곱합
    #     # 예) n=1, 그래디언트 100개의 제곱 합
    #     gradient_sum_timeslot[m]=((sketch_to_vector[m]**2).sum(axis=1))
    #
    #
    #
    # """"# gamma 구하기 식(20)"""
    # gamma=[]
    # for m in range(len(w)):
    #     gamma.append((np.sqrt(sampling.sample([1]))) * (np.sqrt(po / gradient_sum_timeslot[m]) / np.sqrt(6.33153936413615)))
    #
    #
    # """# gamma 평균구하기: 식 (21)"""
    # gamma_avg=torch.zeros(1)
    # for m in range(len(w)):
    #     gamma_avg += gamma[m]
    # gamma_avg=gamma_avg/len(w)
    #######################################################################################################################

    gradient_sum_timeslot = {}
    sketch_to_vector = {}

    ###################################################################################################################
    """기존 N= 여러개일때 방식"""
    for m in range(len(w)):
        # sketchmatrix(rr,cc) -> vector(1,2T)

        sketch_to_vector[m] = ((Sketch[m].table).reshape(1, 2 * T))
        # 제곱 -> (1,2T) -> (2,T) 만들어서 더하기 -> (1,T) -> (100,50) 변형 -> 가로로 더함. (100,1) 은 타임슬롯별 그래디언트 제곱합
        # 예) n=1, 그래디언트 100개의 제곱 합
        gradient_sum_timeslot[m] = (
            (((sketch_to_vector[m] ** 2).reshape(2, T)).sum(axis=0)).reshape(int(T / 50), 50)).sum(axis=1)

    """"# gamma 구하기 식(20)"""
    gamma = []

    for m in range(len(w)):
        for i in range(int(T / 50)):  # Number of timeslot
            gamma.append((torch.sqrt(sampling.sample([1]).to(device))) * (
                        torch.sqrt(po / gradient_sum_timeslot[m][i]) / np.sqrt(6.33153936413615)))

    # gamma list -> gamma matrix로 변형
    gamma_matrix = torch.tensor(gamma).reshape(len(w), int(T / 50)).to(device)

    # gamma_t를 타임슬롯별로 복사하기
    # time slot n= 1,      2,  ....       100
    # gamma_t   [1-50]  [51-100]     [4950-5000]  각 타임슬롯별 동일 감마값 적용
    # 5000size의 감마벡터를 두개를 가로로 연결하여 10000size 벡터로 만들기

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
    ###################################################################################################################

    """# Received signal 구하기: 식(22)"""
    received_signal = torch.zeros(1, 2 * T).to(device)
    for m in range(len(w)):
        received_signal += gamma_vector_cat[m] * sketch_to_vector[m] * sign_matrix[m]

    # 가우시안 노이즈 더하기
    received_signal += ((torch.randn(1, 2 * T).to(device)) / np.sqrt(2))

    """# 감마값 나눠주고, 평균구하기: 식(23) 식(24)"""
    received_signal = received_signal / sum_sign_matrix
    Gcomp_avg = received_signal / (gamma_avg)

    # vec_to_matrix
    Gcomp_avg_matrix = Gcomp_avg.reshape(rr, cc)

    #########################################################################################
    """기존 방식"""
    ##########################################################################################
    """ 모멘텀 주기: 동일 실험 조건을 위해 모멘텀 빼기"""

    if add_momentum:
        s_momentum = 0.9 * s_momentum + Gcomp_avg_matrix

    else:
        s_momentum = Gcomp_avg_matrix

    """ 모든 값 복원하기"""
    Tensor_Gcomp = s_momentum  # 다시 텐서로 변환
    vals = torch.zeros(rr, dd).to(device)
    for r in range(rr):
        vals[r] = (Tensor_Gcomp[r, Sketch_0.buckets[r, :]] * Sketch_0.signs[r, :])
    Recovery_G = vals.median(dim=0)[0]

    # 에러 누적하기
    Recovery_G += g_ec

    """ Top-k 적용"""
    return_topk = torch.zeros_like(Recovery_G)
    _, topkIndices = torch.topk(Recovery_G ** 2, k=K)  #

    return_topk[topkIndices] = Recovery_G[topkIndices]

    """ error accumulation"""
    # 새로운 에러 누적시키기
    # error_accumulation=Recovery_G - return_topk

    # 뺄셈대신, 0으로 만들기
    Recovery_G[topkIndices] = 0
    g_ec = Recovery_G

    """weight 업데이트"""
    Fw_avg = copy.deepcopy(D)
    Fw_avg -= 0.01 * return_topk

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

    # #### #### #### #### #### #### #### #### #### #### #### #### ####
    # """### 새로운 방식 FetchSGD와  똑같이 구현하기 ###"""
    # ####  #### #### #### #### #### #### #### #### #### #### ####
    #
    #
    #
    # """ 모멘텀 주기 """
    # s_momentum= 0.9*s_momentum +Gcomp_avg
    #
    # """에러 더하기"""
    # result=s_momentum+g_ec
    #
    #
    #
    # """ 모든 값 복원하기"""
    # Tensor_Gcomp = torch.Tensor(result)  # 다시 텐서로 변환
    # vals = torch.zeros(rr, dd)
    # for r in range(rr):
    #     vals[r] = (Tensor_Gcomp[r, Sketch_0.buckets[r, :]] * Sketch_0.signs[r, :])
    # Recovery_G = vals.median(dim=0)[0]
    #
    # """ Top-k 적용"""
    # return_topk = torch.zeros_like(Recovery_G)
    # _, topkIndices = torch.topk(Recovery_G ** 2, k=K)  # k:임의로 500으로 설정해보았음
    # return_topk[topkIndices] = Recovery_G[topkIndices]
    #
    #
    #
    #
    # # 복원한 값을 다시 스케칭하기
    # Sketch_g_ec = CSVec(dd, cc, rr, device="cpu")
    # Sketch_g_ec.accumulateVec(return_topk)
    #
    #
    #
    # # 에러 누적하기
    # g_ec=(result)-Sketch_g_ec.table.numpy()

    # """weight 업데이트"""
    # Fw_avg = copy.deepcopy(D)
    # for k in range(0, d):
    #     Fw_avg[k] -= 0.01*return_topk[k]
    #
    # """ flatten -> 매트릭스"""
    # W_avg = recover_flattened(Fw_avg, L, model)
    #
    # """ 모델 매트릭스에 값 넣기"""
    # w_avg = copy.deepcopy(w[0])
    # j = 0
    # for k in w_avg.keys():
    #     w_avg[k] = W_avg[j]
    #     j += 1
    #
    # return w_avg, g_ec, s_momentum


""""###########################################################################################"""
""""###########################################################################################"""

""""###########################################################################################"""
""""###########################################################################################"""


def FedAnalogAMP(w, previous_w_global, T, P, g_ec, po):
    # 모델의 shape 저장하기 "conv1.weight", "conv1.bias" ... ... ... 등
    # [torch.sizie([10,1,5,5]), torch.size([10]), torch.size([20,10,5,5]) ... ]
    model = []

    for k in w[0].keys():
        model.append(w[0][k].shape)

    # W에 모든 파라미터 값을 리스트로 저장한다.

    W = []
    for k in w[0].keys():
        W.append(w[0][k])

    # 플래튼시키기
    D, L = flatten_params(np.array(W))
    d = len(D)

    ##############################################################
    G = np.zeros((len(w), d))
    OG = np.zeros((len(w), d))
    Gprime = np.zeros(len(w))
    Sum = np.zeros(len(w))
    rho = 2 * T / 2
    ##############################################################
    Fw = np.zeros((len(w), d))

    """새로 업데이트된 웨이트 값 플래튼 시키기"""
    for i in range(0, len(w)):
        W = []
        for k in w[0].keys():
            W.append(w[i][k])
        D, L = flatten_params(np.array(W))

        aa = D.to(device='cpu').reshape(21840)
        Fw[i] = aa
        # for k in range(0,d):
        #     Fw[i][k]=D[k]

    """이전 글로벌 웨이트 값 플래튼 시키기"""
    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])

    # D,L=flatten_params(np.array(W))
    D, L = flatten_params(W)

    """ Gradient 구하기"""
    aa = D.to(device='cpu').reshape(21840)
    for i in range(0, len(w)):
        G[i] = (aa - Fw[i]) / 0.01 + g_ec[i]
        OG[i] = G[i]

        # for k in range(0,d):
        #     G[i][k]=(D[k]-Fw[i][k])/0.01+g_ec[i][k]   # error acc 적용
        #     OG[i][k]=G[i][k]

    ###############################################################

    zmean1 = 0
    zvar1 = 0

    """Sparsification"""
    for i in range(0, len(w)):
        Gprime[i] = np.sort(np.abs(G[i][:]))[::-1][int(rho) - 1]  # int(rho) = 2*T/2= 5000
        # gradient를 절대값으로 취한 뒤, 큰값부더 작은갑 순으로 내림 차순으로 정리하고
        # 5000번째 값을 저장한다. (4999번)

    # 5000번째 값(중위값) 보다 작은 그래디언트는 0으로 만든다.

    for i in range(0, len(w)):
        small = np.abs(G[i]) < Gprime[i]
        G[i][small] = 0
        g_ec[i] = OG[i] - G[i]
        zmean1 += G[i]

    zmean1 = zmean1.sum()

    # for k in range(0, d):
    #     if np.abs(G[i][k])<Gprime[i]:
    #         G[i][k] = 0
    #     g_ec[i][k]=OG[i][k]-G[i][k]
    #
    #     #zmean1: 모든 유저의 그래디언트 값을 다 더하기 (0으로 변경된것 포함)
    #     zmean1+=G[i][k]

    # 모든 유저의 그래디언트 더한 값/ 5000 = 각 유저별 그래디언트 평균값 들의 합
    zmean1 = zmean1 / rho / len(w)
    # zmean1=zmean1/rho

    # SS: 모든 유저의 그래디언트 값 더하기
    # SSS: 모든 유저의 그래디언트 제곱값 더하기
    for i in range(0, len(w)):
        SS = G[i].sum()
        SSS = (G[i] ** 2).sum()

        # SS=0
        # SSS=0
        # for k in range(0, d):
        #     SS+=G[i][k]
        #     SSS+=G[i][k]**2

        # 분산 합
        zvar1 += (SSS / rho / (len(w) ** 2)) - (SS / rho / len(w)) ** 2
        # zvar1+=SSS/rho-(SS/rho)**2

    ###################################################################################

    """컴프레싱하기 """
    # 평균0, 분산: np.sqrt(d) 인 가우시안 분포를 가지는 (2T X d)사이즈의 임의 매트릭스 생성
    A = np.random.normal(0, np.sqrt(d), (2 * T, d))

    # A매트릭스와 곱하여서 압축하기  d개(21840) -> 2T개(10000개)
    Gcomp = np.zeros((len(w), 2 * T))
    for i in range(0, len(w)):
        Gcomp[i] = np.dot(A, G[i])
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

    h2_matrix = {}  # 50 by 100 매트릭스
    s_matrix = {}  # 50 by 100 매트릭스
    sign_matrix = {}  # 100 by 100 매트릭스

    sampling = torch.distributions.exponential.Exponential(1.0)

    for m in range(len(w)):
        h2_matrix[m] = sampling.sample([1, T])
        # h^2의 매트릭스 구하기 (50 x 100) : real값과 imag 값 둘다 같은 페이딩 받기때문에, rows=50으로 설정
        # h_matrix[m]=np.zeros((50,100))
        # h_matrix[m] += np.random.exponential(1, (50,100))

        # h^2 > 0.001 적용, 조건보다 작은값은 0으로 만들기
        s_matrix[m] = h2_matrix[m] > 0.001
        vector = np.vectorize(np.int)  # 브로드캐스팅을 위한 함수 설정
        s_matrix[m] = vector(s_matrix[m])  # 1,0으로 표시

        # s_matrix 복제해서 가로로 연결하기: 같은 페이딩 겪게 하기
        sign_matrix[m] = np.concatenate([s_matrix[m], s_matrix[m]], axis=1)

    # sign_matrix의 합 구하기: 차후 나눠주기 위해서
    sum_sign_matrix = np.zeros([1, 2 * T])
    for m in range(len(w)):
        sum_sign_matrix += sign_matrix[m]

    # # gradient l2 구학
    # gradient_sum_timeslot={}
    # for m in range(len(w)):
    #     gradient_sum_timeslot[m]=(Gcomp[m]**2).sum()

    gradient_sum_timeslot = {}
    for m in range(len(w)):
        gradient_sum_timeslot[m] = ((((Gcomp[m] ** 2).reshape(2, T)).sum(axis=0)).reshape(int(T / 50), 50)).sum(axis=1)

    # E_1 구하기
    E_1 = 6.33153936413615
    # 4.726095458584443

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
    # gamma 구하기
    gamma = []
    for m in range(len(w)):
        for i in range(int(T / 50)):
            gamma.append((np.sqrt(np.random.exponential(1, (1, 1)))) * (
            (np.sqrt(po / gradient_sum_timeslot[m][i]) / np.sqrt(E_1))))

    gamma_matrix = np.array(gamma).reshape(len(w), int(T / 50))

    # timeslot 구간 별 gamma 평균 구하기
    # gamma_avg=(gamma_t_array.sum(axis=0))/len(w)

    gamma_vec_cat = {}
    for m in range(len(w)):
        gamma_vec_T = np.zeros([int(T / 50), 50])
        for r in range(int(T / 50)):
            gamma_vec_T[r] = gamma_matrix[m][r]
        gamma_vec = gamma_vec_T.reshape(1, T)
        gamma_vec_cat[m] = np.concatenate([gamma_vec, gamma_vec], axis=1)

    # gamma 평균구하기
    gamma_avg = np.zeros([1, 2 * T])
    for m in range(len(w)):
        gamma_avg += gamma_vec_cat[m]
    gamma_avg = gamma_avg / len(w)
    #####################################################################################################################

    # Received signal 구하기
    received_signal = np.zeros([1, 2 * T])

    for m in range(len(w)):
        received_signal += gamma_vec_cat[m] * Gcomp[m] * sign_matrix[m]
        # received_signal += (gamma_t_array[m].reshape(1,100))*(Gcomp_matrix[m])*(sign_matrix[m])

    # 노이즈 더하기
    received_signal += ((np.random.normal(0, 1, (1, 2 * T))) / np.sqrt(2))

    # 인원수로 나누고, 감마평균값으로 나누기
    received_signal = received_signal / sum_sign_matrix
    Gcomp_avg = received_signal / (gamma_avg)

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
    # AMP apply   & divide by len(w)  Gcomp_avg -->  recovery --> divide by len(w) -->G_avg

    # start=time.time()

    ny = 2 * T
    nz = d
    wvar = 0
    # wvar = 1 / 2 / (gamma_avg_all ** 2) # 노이즈 어떻게 될지..
    # wvar=1/2/(gamma_**2)
    zshape = (nz,)  # Shape of z matrix
    yshape = (ny,)  # Shape of y matrix
    Ashape = (ny, nz)  # Shape of A matrix

    est0 = vp.estim.DiscreteEst(0, 1, zshape)
    est1 = vp.estim.GaussEst(zmean1, zvar1, zshape)

    est_list = [est0, est1]
    prob_on = rho / d
    pz = np.array([1 - prob_on, prob_on])
    est_in = vp.estim.MixEst(est_list, w=pz, name='Input')
    Aop = vp.trans.MatrixLT(A, zshape)
    est_out = vp.estim.LinEst(Aop, Gcomp_avg[0], wvar, map_est=False, name='Output')
    msg_hdl = vp.estim.MsgHdlSimp(map_est=False, shape=zshape)

    nit = 20  # number of iterations
    solver = vp.solver.Vamp(est_in, est_out, msg_hdl, hist_list=['zhat', 'zhatvar'], nit=nit)
    solver.solve()
    G_avg = solver.zhat

    # G_avg=G_avg/len(w) 복원전에 나눴기 때문에 지웠음....

    ##########################################################################################

    "웨이트 업데이트"
    # Fw_avg=copy.deepcopy(D.to(device='cpu'))
    Fw_avg = copy.deepcopy(aa)  # D의 Cpu할당버전= aa

    Fw_avg -= 0.01 * G_avg

    # for k in range(0, d):
    #     Fw_avg[k]-=0.01*G_avg[k]
    #     # 모멘텀 주기
    #     # Fw_avg[k] = 0.9*Fw_avg[k] - 0.01*G_avg[k]

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


def Fed_DDSGD(w, previous_w_global, T, P, g_ec, po, device):
    """############"""
    """채널 관련 처리"""
    # h^2 샘플링
    # 전체 사용할수 있는 서브캐리어는 50개로 고정한다.
    # 한사람이 사용할 수 있는 총 파워 할당(아날로그 기준): po*N
    # 전체 T=5000이면, 한사람당 T=100 ->  서브캐리어 50개* 타임슬롯 2개: T=100, 한사람당 파워: 50*100=5000, 타임슬롯당 2500
    # 전체 T=2500이면, 한사람당 T=50 ->  서브캐리어 50개* 타임슬롯 1개: T=50,    한사람당 파워: 50*50=2500, 타임슬롯당 2500
    # 전체 T=7500이면, 한사람당 T=150 ->  서브캐리어 50개* 타임슬롯 3개: T=150, 한사람당 파워: 50*150=7500, 타임슬롯당 2500
    subcarrier = 50
    m = torch.distributions.exponential.Exponential(1.0)
    h2_matrix = m.sample([int(T / subcarrier), subcarrier]).to(device)  # T5000기준, 100*50 매트릭스

    # 각 유저에게 파워 할당하기
    tol = 1e-2
    p_constrain = po * 100 / 2  # po 50일때. 한사람당 파워 5000할당, 한사람당 타임슬롯별 2500 // po30, 한사람당 파워 3000, 타임슬롯별 1500
    number_vector = h2_matrix[0].shape[0]
    all_p_allocation = torch.zeros([int(T / subcarrier), subcarrier]).to(device)

    for i in range(int(T / subcarrier)):
        noise_vec = 1 / h2_matrix[i]
        water_line = min(noise_vec) + p_constrain / number_vector  # 초기 워터라인 설정
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

        all_p_allocation[i] += p_allocation

    # 각 유저와 중앙서버 간의 capacity 구하기
    p_allo_h2_matrix = all_p_allocation * h2_matrix
    p_allo_h2_matrix = p_allo_h2_matrix.reshape(len(w), -1)  # 사람수로 매트릭스 조정
    p_allo_h2_matrix += 1
    capacitylog = torch.log2(p_allo_h2_matrix)
    capacity = capacitylog.sum(axis=1)

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

        D = D.reshape(21840)
        Fw[i] = D

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
        G[i] = (D - Fw[i]) / 0.01 + g_ec[i]
        OG[i] = G[i]

    """ Gradient 구하기"""  # 넘나 느린방식
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
    all_number_qt = torch.zeros(len(w)).to(device)
    for i in range(len(w)):
        number_qt = 1
        while nCr(d, number_qt) < 2 ** (int(capacity[i]) - 33):
            number_qt += 1
        # 최종 q_t 개수: while문에서 마지막에 1 더해진 값 뺴기
        number_qt -= 1
        all_number_qt[i] = number_qt

    # 유저별 q_t 구하기
    qt_gradient = torch.zeros([len(w), 21840]).to(device)
    index_info = torch.zeros([21840]).to(device)

    # qt 개수가지고 gradient 처리하기
    for i in range(len(w)):
        positive_qt, index_po = torch.topk(OG[i], int(all_number_qt[i]), largest=True)
        negative_qt, index_ne = torch.topk(OG[i], int(all_number_qt[i]), largest=False)
        positive_mean = torch.mean(positive_qt)
        negative_mean = abs(torch.mean(negative_qt))

        if positive_mean > negative_mean:
            qt_gradient[i][index_po] = positive_mean
            # index_info[index_po] += 1
        else:
            qt_gradient[i][index_ne] = torch.mean(negative_qt)
            # index_info[index_ne]+=1

        # error 누적하기
        g_ec[i] = OG[i] - qt_gradient[i]

    # 0인 값에 대해서는 1을 더해주기 (나눠주기 위해서)
    # index_info[torch.nonzero(index_info==0)]+=1

    # q_t avg값 구하기
    qt_avg = (qt_gradient.sum(axis=0)) / len(w)

    """weight 업데이트"""
    Fw_avg = copy.deepcopy(D)
    Fw_avg -= 0.01 * qt_avg

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


def sign_sgd(w, previous_w_global, T, P, g_ec, po, device):
    """############"""
    """채널 관련 처리"""
    # h^2 샘플링
    # 전체 사용할수 있는 서브캐리어는 50개로 고정한다.
    # 한사람이 사용할 수 있는 총 파워 할당(아날로그 기준): po*N
    # 전체 T=5000이면, 한사람당 T=100 ->  서브캐리어 50개* 타임슬롯 2개: T=100, 한사람당 파워: 50*100=5000, 타임슬롯당 2500
    # 전체 T=2500이면, 한사람당 T=50 ->  서브캐리어 50개* 타임슬롯 1개: T=50,    한사람당 파워: 50*50=2500, 타임슬롯당 2500
    # 전체 T=7500이면, 한사람당 T=150 ->  서브캐리어 50개* 타임슬롯 3개: T=150, 한사람당 파워: 50*150=7500, 타임슬롯당 2500
    subcarrier = 50
    m = torch.distributions.exponential.Exponential(1.0)
    h2_matrix = m.sample([int(T / subcarrier), subcarrier]).to(device)  # T5000기준, 100*50 매트릭스

    # 각 유저에게 파워 할당하기
    tol = 1e-2
    p_constrain = po * 100 / 2  # po 50일때. 한사람당 파워 5000할당, 한사람당 타임슬롯별 2500 // po30, 한사람당 파워 3000, 타임슬롯별 1500
    number_vector = h2_matrix[0].shape[0]
    all_p_allocation = torch.zeros([int(T / subcarrier), subcarrier]).to(device)

    for i in range(int(T / subcarrier)):
        noise_vec = 1 / h2_matrix[i]
        water_line = min(noise_vec) + p_constrain / number_vector  # 초기 워터라인 설정
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

        all_p_allocation[i] += p_allocation

    # 각 유저와 중앙서버 간의 capacity 구하기
    p_allo_h2_matrix = all_p_allocation * h2_matrix
    p_allo_h2_matrix = p_allo_h2_matrix.reshape(len(w), -1)  # 사람수로 매트릭스 조정
    p_allo_h2_matrix += 1
    capacitylog = torch.log2(p_allo_h2_matrix)
    capacity = capacitylog.sum(axis=1)

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
        D = D.reshape(21840)
        Fw[i] = D

    """ 직전의 글로벌 웨이트 값 플래튼 시키기"""
    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])
    D, L = flatten_params((W))

    """ Gradient 구하기"""
    D = D.reshape(21840)
    for i in range(len(w)):
        G[i] = (D - Fw[i]) / 0.01 + g_ec[i]
        OG[i] = G[i]

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

    all_number_qt = torch.zeros(len(w)).to(device)
    for i in range(len(w)):
        number_qt = 1
        while ((nCr(d, number_qt)) - (
                2 ** (int(capacity[i]) - number_qt))) < 0:  # capacity를 int형으로 변환해도 되는지? infinity 막기 위해서: 결론 별차이 없다.
            number_qt += 1
        # 최종 q_t
        number_qt -= 1
        all_number_qt[i] = number_qt

    # 유저별 qt 구하기
    qt_gradient = torch.zeros([len(w), 21840]).to(device)
    index_info = torch.zeros([21840]).to(device)

    # TOPK 값 받기

    for i in range(len(w)):
        q_t, index_qt = torch.topk(abs(OG[i]), int(all_number_qt[i]), largest=True)

        sign_G = torch.zeros(int(all_number_qt[i])).to(device)

        # 0보다 크면 1, 0보다 작으면 -1로 설정하기
        for j in range(int(all_number_qt[i])):
            if OG[i][index_qt][j] > 0:
                sign_G[j] = 1
            else:
                sign_G[j] = -1

        # index_info[index_qt]+=1
        # 인덱스에 sign_G값 넣고, 나머지 ZERO화
        qt_gradient[i][index_qt] = sign_G

        # error 누적하기
        g_ec[i] = OG[i] - qt_gradient[i]

    # index_info[torch.nonzero(index_info==0)]+=1

    # qt_avg=(qt_gradient.sum(axis=0))/index_info
    qt_avg = (qt_gradient.sum(axis=0)) / len(w)

    """weight 업데이트"""
    Fw_avg = copy.deepcopy(D)
    Fw_avg -= 0.01 * qt_avg

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


def FedAnalogRandom(w, previous_w_global, T, P, g_ec):
    model = []

    for k in w[0].keys():
        model.append(w[0][k].shape)

    W = []
    for k in w[0].keys():
        W.append(w[0][k])
    D, L = flatten_params(np.array(W))
    d = len(D)
    ##############################################################
    G = np.zeros((len(w), d))
    OG = np.zeros((len(w), d))
    Gprime = np.zeros(len(w))
    Sum = np.zeros(len(w))
    ##############################################################
    Fw = np.zeros((len(w), d))

    for i in range(0, len(w)):
        W = []
        for k in w[0].keys():
            W.append(w[i][k])
        D, L = flatten_params(np.array(W))
        for k in range(0, d):
            Fw[i][k] = D[k]

    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])
    D, L = flatten_params(np.array(W))

    for i in range(0, len(w)):
        for k in range(0, d):
            G[i][k] = (D[k] - Fw[i][k]) / 0.01 + g_ec[i][k]
            OG[i][k] = G[i][k]

    ###############################################################

    Sp = np.random.choice(d, 2 * T, replace=False)
    np.sort(Sp)
    SP = np.zeros(d)
    G_avg = np.zeros(d)

    for i in range(0, 2 * T):
        SP[Sp[i]] = 1

    for i in range(0, len(w)):
        for k in range(0, d):
            if SP[k] == 0:
                G[i][k] = 0
            Sum[i] += G[i][k] ** 2
            g_ec[i][k] = OG[i][k] - G[i][k]

    gamma = np.sqrt(np.random.exponential(1, (1, 1))) * np.sqrt(P / Sum[0])
    for i in range(1, len(w)):
        gamma = min(gamma, np.sqrt(np.random.exponential(1, (1, 1))) * np.sqrt(P / Sum[i]))

    for k in range(0, d):
        for i in range(0, len(w)):
            if SP[k] == 1:
                G_avg[k] += G[i][k]

        G_avg[k] + np.random.normal(1, 0, (1, 1)) / gamma / np.sqrt(2)

    G_avg = G_avg / len(w)

    ##################################################################################

    Fw_avg = copy.deepcopy(D)
    for k in range(0, d):
        Fw_avg[k] -= 0.01 * G_avg[k]

    W_avg = recover_flattened(Fw_avg, L, model)
    w_avg = copy.deepcopy(w[0])

    j = 0
    for k in w_avg.keys():
        w_avg[k] = W_avg[j]
        j += 1

    return w_avg, g_ec


def FedAnalogPSGuide(w, previous_w_global, T, P, g_ec):
    ##############################################Add a devcie to 0th device

    model = []

    for k in w[0].keys():
        model.append(w[0][k].shape)

    W = []
    for k in w[0].keys():
        W.append(w[0][k])
    D, L = flatten_params(np.array(W))
    d = len(D)
    ##############################################################
    G = np.zeros((len(w), d))
    OG = np.zeros((len(w), d))
    Sum = np.zeros(len(w))
    ##############################################################
    Fw = np.zeros((len(w), d))

    for i in range(0, len(w)):
        W = []
        for k in w[0].keys():
            W.append(w[i][k])
        D, L = flatten_params(np.array(W))
        for k in range(0, d):
            Fw[i][k] = D[k]

    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])
    D, L = flatten_params(np.array(W))

    for i in range(0, len(w)):
        for k in range(0, d):
            G[i][k] = (D[k] - Fw[i][k]) / 0.01 + g_ec[i][k]
            OG[i][k] = G[i][k]

    ###############################################################

    Gprime = np.sort(np.abs(G[0]))[::-1][2 * T - 1]
    SP = np.ones(d)
    G_avg = np.zeros(d)

    for k in range(0, d):
        if np.abs(G[0][k]) < Gprime:
            SP[k] = 0

    #################################################################

    for i in range(1, len(w)):
        for k in range(0, d):
            if SP[k] == 0:
                G[i][k] = 0
            Sum[i] += G[i][k] ** 2
            g_ec[i][k] = OG[i][k] - G[i][k]

    gamma = np.sqrt(np.random.exponential(1, (1, 1))) * np.sqrt(P / Sum[1])
    for i in range(2, len(w)):
        gamma = min(gamma, np.sqrt(np.random.exponential(1, (1, 1))) * np.sqrt(P / Sum[i]))

    for k in range(0, d):
        for i in range(1, len(w)):
            if SP[k] == 1:
                G_avg[k] += G[i][k]

        G_avg[k] + np.random.normal(1, 0, (1, 1)) / gamma / np.sqrt(2)

    G_avg = G_avg / (len(w) - 1)

    ##################################################################################

    Fw_avg = copy.deepcopy(D)
    for k in range(0, d):
        Fw_avg[k] -= 0.01 * G_avg[k]

    W_avg = recover_flattened(Fw_avg, L, model)
    w_avg = copy.deepcopy(w[0])

    j = 0
    for k in w_avg.keys():
        w_avg[k] = W_avg[j]
        j += 1

    return w_avg, g_ec


def FedAnalogOneDeviceGuide(w, previous_w_global, T, P, g_ec, r):
    model = []

    for k in w[0].keys():
        model.append(w[0][k].shape)

    W = []
    for k in w[0].keys():
        W.append(w[0][k])
    D, L = flatten_params(np.array(W))
    d = len(D)
    ##############################################################
    G = np.zeros((len(w), d))
    OG = np.zeros((len(w), d))
    Sum = np.zeros(len(w))
    ##############################################################
    Fw = np.zeros((len(w), d))

    for i in range(0, len(w)):
        W = []
        for k in w[0].keys():
            W.append(w[i][k])
        D, L = flatten_params(np.array(W))
        for k in range(0, d):
            Fw[i][k] = D[k]

    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])
    D, L = flatten_params(np.array(W))

    for i in range(0, len(w)):
        for k in range(0, d):
            G[i][k] = (D[k] - Fw[i][k]) / 0.01 + g_ec[i][k]
            OG[i][k] = G[i][k]

    ###############################################################

    T1 = int(r * T)
    K = 0

    h = np.sqrt(np.random.exponential(1, (len(w), 1)))
    H = h[0]
    ii = 0
    for i in range(1, len(w)):
        if h[i] > H:
            H = h[i]
            ii = i

    SP = np.ones(d)

    start = time.time()

    sm = 0
    for k in range(1, T):
        sm += np.log2(d - 2 * (k - 1) - 1 + 1)
        sm += np.log2(d - 2 * (k - 1) - 2 + 1)
        sm -= np.log2(2 * (k - 1) + 1)
        sm -= np.log2(2 * (k - 1) + 2)
        if T1 * np.log2(1 + (H ** 2) * P / T) < sm:
            K = k - 1
            break
    period = time.time() - start

    if K == 0:
        Gprime = 10 ** 6
        Sp = np.random.choice(d, 2 * T, replace=False)
        np.sort(Sp)
        SP = np.zeros(d)
        for i in range(0, 2 * T):
            SP[Sp[i]] = 1
    else:
        if T - T1 - K > 0:
            Sp = np.random.choice(d, 2 * (T - T1 - K), replace=False)
            np.sort(Sp)
            SP = np.zeros(d)
            I = np.ones(d)

            for i in range(0, 2 * (T - T1 - K)):
                SP[Sp[i]] = 1

            G_temp = G[ii] * (I - SP)
            Gprime = np.sort(np.abs(G_temp))[::-1][2 * K - 1]

        else:
            SP = np.zeros(d)
            Gprime = np.sort(np.abs(G[ii]))[::-1][2 * (T - T1) - 1]
    #######################################################################

    for i in range(0, len(w)):
        for k in range(0, d):
            if SP[k] == 0 and np.abs(G[ii][k]) < Gprime:
                G[i][k] = 0
            Sum[i] += G[i][k] ** 2
            g_ec[i][k] = OG[i][k] - G[i][k]

    if ii == 0:
        gamma = h[0] * np.sqrt((T - T1) * P / Sum[0] / T)
    else:
        gamma = h[0] * np.sqrt(P / Sum[0])

    for i in range(1, len(w)):
        if ii == i:
            gamma = min(gamma, h[i] * np.sqrt((T - T1) * P / Sum[i] / T))
        else:
            gamma = min(gamma, h[i] * np.sqrt(P / Sum[i]))

    G_avg = np.zeros(d)
    for k in range(0, d):
        for i in range(0, len(w)):
            G_avg[k] += G[i][k]

        G_avg[k] + np.random.normal(1, 0, (1, 1)) / gamma / np.sqrt(2)

    G_avg = G_avg / len(w)

    ##################################################################################

    Fw_avg = copy.deepcopy(D)
    for k in range(0, d):
        Fw_avg[k] -= 0.01 * G_avg[k]

    W_avg = recover_flattened(Fw_avg, L, model)
    w_avg = copy.deepcopy(w[0])

    j = 0
    for k in w_avg.keys():
        w_avg[k] = W_avg[j]
        j += 1

    return w_avg, g_ec


def FedAnalogAllDeviceGuide(w, previous_w_global, T, P, g_ec):
    model = []
    for k in w[0].keys():
        model.append(w[0][k].shape)

    W = []
    for k in w[0].keys():
        W.append(w[0][k])
    D, L = flatten_params(np.array(W))
    d = len(D)
    ##############################################################
    G = np.zeros((len(w), d))
    OG = np.zeros((len(w), d))
    Sum = np.zeros(len(w))
    ##############################################################
    Fw = np.zeros((len(w), d))

    for i in range(0, len(w)):
        W = []
        for k in w[0].keys():
            W.append(w[i][k])
        D, L = flatten_params(np.array(W))
        for k in range(0, d):
            Fw[i][k] = D[k]

    W = []
    for k in w[0].keys():
        W.append(previous_w_global[k])
    D, L = flatten_params(np.array(W))

    for i in range(0, len(w)):
        for k in range(0, d):
            G[i][k] = (D[k] - Fw[i][k]) / 0.01 + g_ec[i][k]
            OG[i][k] = G[i][k]

    d1 = math.ceil(d / len(w))
    d2 = d - d1 * (len(w) - 1)
    h = np.random.exponential(1, (len(w), 1))
    H = h[0]

    ii = 0
    for i in range(1, len(w)):
        if H > h[i]:
            H = h[i]
            ii = i

    T1 = 0
    sm = 0
    for k in range(1, math.floor(T / len(w))):
        if ii == len(w) - 1:
            sm += np.log2(d2 - 2 * (k - 1) - 1 + 1)
            sm += np.log2(d2 - 2 * (k - 1) - 2 + 1)
            sm -= np.log2(2 * (k - 1) + 1)
            sm -= np.log2(2 * (k - 1) + 2)
        else:
            sm += np.log2(d1 - 2 * (k - 1) - 1 + 1)
            sm += np.log2(d1 - 2 * (k - 1) - 2 + 1)
            sm -= np.log2(2 * (k - 1) + 1)
            sm -= np.log2(2 * (k - 1) + 2)

        if (math.floor(T / len(w)) - k) * np.log2(1 + (H ** 2) * P * len(w) / T) < sm:
            T1 = math.floor(T / len(w)) - (k - 1)
            break

    Gprime = np.zeros(len(w))

    for i in range(0, len(w) - 1):
        if T1 == math.floor(T / len(w)):
            Gprime[i] = 10 ** 6
        else:
            Ghat = np.zeros(d1)
            for aa in range(d1 * i, d1 * (i + 1)):
                Ghat[aa - d1 * i] = G[i][aa]
            Gprime[i] = np.sort(np.abs(Ghat))[::-1][2 * (math.floor(T / len(w)) - T1) - 1]

    Ghat = np.zeros(d - d1 * (len(w) - 1))

    if len(w) > 1:
        if T1 == math.floor(T / len(w)):
            Gprime[i] = 10 ** 6
        else:
            for aa in range(d1 * (len(w) - 1), d):
                Ghat[aa - d1 * (len(w) - 1)] = G[len(w) - 1][aa]
            Gprime[len(w) - 1] = np.sort(np.abs(Ghat))[::-1][2 * (math.floor(T / len(w)) - T1) - 1]

    SP = np.ones(d)

    for i in range(0, len(w) - 1):
        for k in range(d1 * i, d1 * (i + 1)):
            if np.abs(G[i][k]) < Gprime[i]:
                SP[k] = 0

    if len(w) > 1:
        for k in range(d1 * (len(w) - 1), d):
            if np.abs(G[len(w) - 1][k]) < Gprime[len(w) - 1]:
                SP[k] = 0

                ###################################################################################

    for i in range(0, len(w)):
        for k in range(0, d):
            if SP[k] == 0:
                G[i][k] = 0
            Sum[i] += G[i][k] ** 2
            g_ec[i][k] = OG[i][k] - G[i][k]

        if T1 == math.floor(T / len(w)):
            Sum[i] = 1

    gamma = h[0] * np.sqrt((T - len(w) * T1) * P / Sum[0] / T)
    for i in range(1, len(w)):
        gamma = min(gamma, h[i] * np.sqrt((T - len(w) * T1) * P / Sum[i] / T))

    if gamma == 0:
        gamma = 1

    G_avg = np.zeros(d)
    for k in range(0, d):
        for i in range(0, len(w)):
            if SP[k] == 1:
                G_avg[k] += G[i][k]

        G_avg[k] + np.random.normal(1, 0, (1, 1)) / gamma / np.sqrt(2)

    G_avg = G_avg / len(w)

    ##################################################################################

    Fw_avg = copy.deepcopy(D)
    for k in range(0, d):
        Fw_avg[k] -= 0.01 * G_avg[k]

    W_avg = recover_flattened(Fw_avg, L, model)
    w_avg = copy.deepcopy(w[0])

    j = 0
    for k in w_avg.keys():
        w_avg[k] = W_avg[j]
        j += 1

    return w_avg, g_ec



