#-*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time
import sympy
from sympy import Integral, Symbol, exp, S
import math

from models.flatten_params import flatten_params
from models.recover_flattened import recover_flattened
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate, DatasetSplit

from models.Nets import MLP, CNNMnist, CNNCifar
from models.Strategy import FedAvg, FedAnalogAMP,Fedtopk, FedSketch, Fed_DDSGD, sign_sgd, sign_major, FedAnalogRandom, FedAnalogPSGuide, FedAnalogOneDeviceGuide
from models.test import test_img
import random


#from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(edgeitems=100)

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    used_index = []

    torch.cuda.set_device(args.device) # change allocation of current GPU
    random_seed=100+1000*args.setting
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    """샘플링 과정"""
    # load dataset and split users
    if args.dataset == 'mnist':
        args.num_channels=1
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        
        # sample users
        if args.iid=='True':
            dict_users = mnist_iid(dataset_train, args.num_users)

            ## Revised: 필요없음.
            # for i in range(len(dict_users)):
            #         tmp = list(dict_users[i])
            #         for j in range(len(dict_users[0])):
            #             used_index.append(tmp[j])
            
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.setting)
            # dict_users[0], dict_users[1] .... 각 300개씩 데이터 부여

            if args.execute == 'psguide':
                PS_data=mnist_iid(dataset_train, 1)
                dict_users[0]=PS_data[0]

            # 학습한 데이터로 테스팅하기 위한 과정: 현재 필요 없음
            # for i in range(len(dict_users)):
            #     tmp = list(dict_users[i])
            #     # tmp는 dict_user[i] 에서 받은것을 리스트화 한다.
            #     for j in range(len(dict_users[0])):
            #         used_index.append(tmp[j])
            #     # used_index: 이번 학습간 사용한 데이터를 모두 저장한다.


    elif args.dataset == 'cifar':
        args.num_channels=3
        trans_cifar = transforms.Compose([transforms.ToTensor(),
             transforms.RandomHorizontalFlip(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=test_trans_cifar)
        if args.iid=='True':
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users, args.setting)
            if args.execute == 'psguide':
                PS_data=mnist_iid(dataset_train, 1)
                dict_users[0]=PS_data[0]
    else:
        exit('Error: unrecognized dataset')













    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    w_size = []
    for k in w_glob.keys():
        w_size.append(w_glob[k]) #갑자기 안됨?


    D, _ = flatten_params(w_size)
    #D, _ = flatten_params(np.array(w_size))
    model_length = len(D)
    



    """initialize g_ec"""
    if args.execute=='sketch':
        w_size = []
        for k in w_glob.keys():
            w_size.append(w_glob[k]) #갑자기 안됨?


        D, _ = flatten_params(w_size)
        #D, _ = flatten_params(np.array(w_size))
        model_length = len(D)
        

        #error_accumulation (기존방식)
        g_ec = torch.zeros(model_length).to(args.device) # model_length개의 벡터
        #g_ec = np.zeros(model_length)

        # # error_accumulation(새로운방식)
        # g_ec=np.zeros([int(args.cc),int((2*args.T/args.cc))])

        # 모멘텀 관련
        s_momentum=np.zeros([int(args.cc),int((2*args.T/args.cc))])


    elif args.execute=="D_DSGD":

        #g_ec=torch.zeros([model_length,1],device="cuda:0")
        g_ec = torch.zeros([args.num_users, model_length]).to(args.device)
    elif args.execute=="avg":
        g_ec = torch.zeros([args.num_users, model_length]).to(args.device)
    elif args.execute=="sign":
        g_ec = torch.zeros([args.num_users, model_length]).to(args.device)

    elif args.execute=="sign_major":
        g_ec = torch.zeros([args.num_users, model_length]).to(args.device)

    elif args.execute=="random":
        g_ec= torch.zeros([args.num_users, model_length]).to(args.device)
    elif args.execute=="topk":
        g_ec= torch.zeros([args.num_users, model_length]).to(args.device)
    elif args.execute=="psguide":
        g_ec= torch.zeros([args.num_users, model_length]).to(args.device)
    elif args.execute=="psguideN":
        g_ec= torch.zeros([args.num_users, model_length]).to(args.device)
    elif args.execute=="onedevice":
        g_ec= torch.zeros([args.num_users, model_length]).to(args.device)
    else:
        W=[]
        for k in w_glob.keys():
            W.append(w_glob[k])
        
        D,L=flatten_params(W)
        d=len(D)
        g_ec=np.zeros((args.num_users,d))






    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    numbernonzero=[]
    # test
    loss_test = []
    acc_test = []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        # w_locals에 유저수만큼, w_glob를 생성함.
        # w_locals[0] = w_glob
        # w_locals[1] = w_glob
        #  .....
        # w_locals[99]= w_glob
        # 글로벌 파라미터들을 각 로컬들에게 전송하는 개념
    args.po=10**(args.po/10)
    for iter in range(args.epochs):


        args.lr*=0.99 ############################################################################################################################
        start = time.time()

        loss_locals = []
        # loss_locals= 최초 공리스트 생성

        if not args.all_clients:
            w_locals = []
            # all_clients가 없으면 w_locals: 공 리스트 설정
            # all_clients가 한번 호출되었다면 그 값을 받는다.

        m = max(int(args.frac * args.num_users), 1)
        # 프랙*유저= 유저 숫자를 선택한다. 디폴트: 0.1*100 = 10명

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # idxs_users: 유저 숫자중에 m명만큼 랜덤하게 선택한다.
        # 0-99 중에서 10개를 선택한다.
        # 예: 1,2,3,4,5,6,7,8,9,10 선택
        # replace=False 비복원추출
        # 디폴트 에포크가 10이므로, 결국 100명의 유저가 학습하게됨. (중복가능)

        #start = time.time()

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            # local: LocalUpdate 클래스로 객체 생성 (args 값 받고, dataset= 전체 트레인데이터 셋, idx= 선택된 유저가 가진 데이트의 인덱스 값
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # net= net_glob를 딥카피하고, 트레인 시킨다.
            # local update에서 트레인 후 return 한 state_dict() 와 로스를 w, loss에 저장한다.
            
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
                # w_locals[idx]는 다시 업데이트 된 w를 딥카피한다.
                # 학습에 참여한 디바이스는 업데이트 된다.

            else:
                w_locals.append(copy.deepcopy(w))
                # all_clients가 호출 안되었다면
                # 현재 공리스트인 w_locals에 w를 딥카피하여 추가한다.
                # w_locals에는 계산한 순서대로 w_locals가 차례로 추가된다.

                # w_locals는 리스트 지만 w_locals[0], w_locals[1] 은 딕셔너리 형태로 웨이트 값이 저장되어있음

                # all_client가 없다면, distributed learning 방식이다.
                # 학습에 참여한 디바이스들끼리 지속적으로 업데이트하는 방식임.

            loss_locals.append(copy.deepcopy(loss))

        #print("time:", time.time() - start)


        """ update global weights """


        # 파라미터 관련
        # P값 받기
        P=args.T * args.po
        # TopK 값 정학
        K=int(args.K)
        cc=int(args.cc)


        if args.execute == 'avg':
            w_glob = FedAvg(w_locals,args.device,args.lr)
        elif args.execute == 'amp':
            w_glob, g_ec = FedAnalogAMP(w_locals, w_glob,args.T,P, g_ec,args.po,args.add_error,args.lr)
        elif args.execute == 'random':
            w_glob, g_ec = FedAnalogRandom(w_locals, w_glob,args.T,P, g_ec,args.po,args.device,args.add_error,args.lr)
        elif args.execute == 'topk':
            w_glob, g_ec = Fedtopk(w_locals, w_glob,args.T,P, g_ec,args.po,args.device,args.add_error,args.lr)
        elif args.execute == 'psguide':
            w_glob, g_ec = FedAnalogPSGuide(w_locals, w_glob,args.T,P, g_ec,args.po,args.device, K, args.add_error,args.lr)
        elif args.execute == 'psguideN':
            w_glob, g_ec = FedAnalogPSGuide(w_locals, w_glob,args.T,P, g_ec,args.po,args.device, K, args.add_error,args.lr)    
        elif args.execute == 'onedevice':
            w_glob, g_ec = FedAnalogOneDeviceGuide(w_locals, w_glob,args.T,P, g_ec,args.po,args.device, K, args.add_error,args.r,args.lr)   
        #elif args.execute == 'alldevice':
        #    w_glob, g_ec = FedAnalogAllDeviceGuide(w_locals, w_glob,args.T,P, g_ec)
        elif args.execute =='sketch':
            w_glob, g_ec, s_momentum = FedSketch(w_locals,w_glob, args.T,P,K,cc,g_ec,s_momentum,args.po,args.add_momentum,args.device,iter,args.cc,args.add_error,args.lr)
            #w_glob, g_ec, s_momentum = FedSketch(w_locals, w_glob, args.T, P, K, cc, g_ec, s_momentum, args.po, args.add_momentum)
        elif args.execute =='D_DSGD':
            w_glob, g_ec = Fed_DDSGD(w_locals, w_glob, args.T, P, g_ec, args.po, args.device,args.lr)
        elif args.execute =='sign':
            w_glob, g_ec = sign_sgd(w_locals, w_glob, args.T, P, g_ec, args.po, args.device,args.lr)
        elif args.execute =='sign_major':
            w_glob, g_ec = sign_major(w_locals, w_glob, args.T, P, g_ec, args.po, args.device,args.lr)
        else:
            exit('Error: unrecognized strategy')

        
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # load_state_dict는 w_glob 값을 불러와서, net_glob에 채운다.
        # w_glob는 각 기법을  통해 업데이트된 값이며, 이것을 net_glob의 파라미터로 업데이트 하는 것임
        if args.execute=='topk':
            model = []

            for k in w_glob.keys():
                model.append(w_glob[k].shape)

            W = []
            for k in w_glob.keys():
                W.append(w_glob[k])
            D, L = flatten_params(W)
            
            numbernonzero.append(sum(torch.tensor(D)!=0))

        # test result per iteration
        test_accuracy, test_loss = test_img(net_glob, dataset_test, args)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        print('Round {:3d}, Test accuracy {:.3f}'.format(iter,test_accuracy))
        loss_train.append(loss_avg)
        loss_test.append(test_loss)
        acc_test.append(test_accuracy)

        print("time:", time.time() - start)
        if math.isnan(test_loss):
            break


    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('Training loss',weight="bold")
    # plt.xlabel('Rounds',weight="bold")
    # plt.axis([0, args.epochs, 0, 100])
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # test_loss to CSV
    loss_train=np.array(loss_train)
    T ="\n".join(map(str, loss_train))
    f = open('./result/loss_{}_iid_{}_{}_T{}_P{}_U{}_Err{}_lr{}_seed{}.csv'.format(args.dataset,args.iid,args.execute, args.T, args.po, args.num_users,args.add_error,args.lr,args.setting),'w')
    f.write(T)
    f.close()
    
    # test_acc to CSV
    acc_test = np.array(acc_test)
    A ="\n".join(map(str, acc_test))
    f = open('./result/acc_{}_iid_{}_{}_T{}_P{}_U{}_Err{}_lr{}_seed{}.csv'.format(args.dataset,args.iid,args.execute, args.T, args.po, args.num_users,args.add_error,args.lr,args.setting),'w')
    f.write(A)
    f.close()

    if args.execute =='topk':
        N ="\n".join(map(str, numbernonzero))
        f = open('./result/numbernonzero_{}_iid_{}_{}_T{}_P{}_U{}_Err{}_lr{}_seed{}.csv'.format(args.dataset,args.iid,args.execute, args.T, args.po, args.num_users,args.add_error,args.lr,args.setting),'w')
        f.write(N)
        f.close()
    # 학습에 참여한 데이터로만 테스트 하기 위한 준비
    select_data_set_idx = set([])
    # set 형태 생성
    for i in range(len(dict_users)):
        select_data_set_idx = select_data_set_idx.union(dict_users[i])
    # 선택된 유저의 인덱스 합치기
    select_data_set = DatasetSplit(dataset_train, select_data_set_idx)


    # 이것은 영준이 버전
    # splitted_dataset = []
    # for i in range(len(used_index)):
    #     splitted_dataset.append(dataset_train[used_index[i]])
    # splitted_dataset = tuple(splitted_dataset)

    
    # testing
    net_glob.eval()
        
    # acc_train, loss_train = test_img(net_glob, splitted_dataset, args)

    acc_train, loss_train = test_img(net_glob, select_data_set, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

