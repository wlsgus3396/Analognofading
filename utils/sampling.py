import numpy as np
import random
import time
from torchvision import datasets, transforms




def mnist_iid(dataset, num_users):
    """
    Fed와 다른점: 유저당 데이터수를 300개만 가진다.
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    print("Start MNIST-IID")
    #import ipdb; ipdb.set_trace()
    #num_items = int(len(dataset)/num_users)
    num_items = 300
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        #all_idxs = list(set(all_idxs) - dict_users[i])
    #global splitted_dataset

    #import ipdb; ipdb.set_trace()
    return dict_users


def mnist_noniid(dataset, num_users,seed):


    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    
    num_shards, num_imgs = 400, 150       
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # random seed 효과 부여
    rng=random.Random(seed)


    # divide and assign
    for i in range(num_users):
        #rand_set1=set(np.random.choice(np.arange(10), 2, replace=False)) # 기존것
        rand_set1=set(rng.sample([0,1,2,3,4,5,6,7,8,9],2))  # random seed 적용

        rand_set=[]
        for rand in rand_set1:  
            #rand_set.extend(np.random.choice(np.arange(40*rand,40*(rand+1)), 1, replace=False))  #기존것
            rand_set.extend(rng.sample(set(np.arange(40*rand,40*(rand+1))),1))  # random seed 적용
        #rand_set= set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in set(rand_set):
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)


    return dict_users








def cifar_iid(dataset, num_users):
    """
    Fed와 다른점: 유저당 데이터수를 300개만 가진다.
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    print("Start CIFAR-IID")
    #import ipdb; ipdb.set_trace()
    #num_items = int(len(dataset)/num_users)
    num_items = 1000#########################################################################
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        #all_idxs = list(set(all_idxs) - dict_users[i])
    #global splitted_dataset

    #import ipdb; ipdb.set_trace()
    return dict_users


def cifar_noniid(dataset, num_users,seed):


    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    
    num_shards, num_imgs = 200, 500       
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    #labels = dataset.targets.numpy()
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # random seed 효과 부여
    rng=random.Random(seed)

    print(seed)


    # divide and assign
    for i in range(num_users):
        #rand_set1=set(np.random.choice(np.arange(10), 2, replace=False)) # 기존것
        rand_set1=set(rng.sample([0,1,2,3,4,5,6,7,8,9],2))  # random seed 적용

        rand_set=[]
        for rand in rand_set1:  
            #rand_set.extend(np.random.choice(np.arange(40*rand,40*(rand+1)), 1, replace=False))  #기존것
            rand_set.extend(rng.sample(set(np.arange(20*rand,20*(rand+1))),1))  # random seed 적용
        #rand_set= set(np.random.choice(idx_shard, 1, replace=False))
        #idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in set(rand_set):
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)


    return dict_users