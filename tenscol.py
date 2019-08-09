import numpy as np
import torch as th
from tqdm import tqdm
import pandas as pd
import datetime
from joblib import Parallel, delayed
import time

""" 
Pytorch implementation of TensCol
Source code of the paper "Gradient descent based learning for grouping problems : Application on Graph Coloring and Equitable Graph Coloring"
Author: Olivier Goudet
Date: 08/08/2019
"""


def job_function(instanceList, D, alpha, beta, lambda_, mu_, nu_, eta_, sigma_0, nb_iter, rho, max_iter=2000000, idx=0):

    index_dataset = int(idx/(nbrun))
    idx = int(idx%nbrun)
    time.sleep(idx)

    instance = instanceList[index_dataset]
    tmp = instance.split("-")
    nameGraph = tmp[0]
    k = int(tmp[1])

    if(tmp[2] == "ECP"):
        isECP = True
    else:
        isECP = False

    print("nameGraph : " + str(nameGraph))
    print("k : " + str(k))

    # get graph
    filename_graph = "benchmark/" + nameGraph + ".csv"
    graph = np.loadtxt(filename_graph, delimiter=",")
    n = int(np.max(graph))

    # create adjacency matrix from graph file
    triangsup = np.zeros((n, n))

    for j in range(graph.shape[0]):
        triangsup[int(graph[j][0]) - 1, int(graph[j][1]) - 1] = 1
    A = triangsup + np.transpose(triangsup)
    n = A.shape[0]


    # Allocate one gpu for the job
    gpu = List_free_device.pop()
    device = "cuda:{}".format(int(gpu))

    start = time.time()
    time_elapsed = -1

    # Adjacency matrix of the graph
    A = th.from_numpy(A.astype('float32')).to(device)

    # Tensor of weights
    W = th.zeros((D, n, k)).to(device)

    # Init weight with normal law and 0.01 standard deviation
    W.data.normal_(0, sigma_0)

    if(isECP):
        c1 = int(n / k)
        c2 = int(n / k) + 1
        zeros = th.zeros((n,k)).to(device)
        ones = th.ones((n,k)).to(device)

    J = th.ones((k, k)).to(device)

    best_score = 99999999

    if (verbose):
        pbar = tqdm(range(max_iter))
    else:
        pbar = range(max_iter)

    for t in pbar:

        # Group selection - build S from W
        logits = W.data
        _, project = W.data.max(-1)
        shape = logits.size()
        S = logits.data.new(*shape).zero_().scatter_(-1, project.view(shape[0], shape[1], 1), 1.0)

        # Compute association matrix V
        V = S @ S.transpose(1, 2)

        # Compute conflict matrix C
        C =  A * V

        # Compute the GCP fitness vector
        f = th.sum(C, dim=[1, 2]) / 2.0

        if (isECP):
            # Compute the equity fitness vector
            count = th.sum(S, dim=1)
            diff_count = th.min(th.abs(count - c1), th.abs(count - c2))
            e = th.sum(diff_count, dim=1)


        # Compute the global loss
        if (isECP):
            loss = f + e
        else:
            loss = f

        # Min of the global loss for the D candidate solutions
        min_loss = th.min(loss)

        # Store best score
        if (min_loss.item() < best_score):
            best_score = min_loss.item()

        if (verbose):
            if t % 10 == 0 and t > 1:
                pbar.set_postfix(min_loss=min_loss.item(), best_score=best_score)


        #Compute GCP gradient
        gamma =  A @ S

        # Compute group concentration matrix
        V_bar = th.sum(V,0)

        # Compute the gradient of the kappa diversity penalization term
        kappa_grad = 2*alpha*lambda_* t *(A*V_bar**(alpha-1)) @ S

        if (isECP):
            # Compute gradient of the equity fitness
            count = th.sum(S, dim=1,keepdim=True)
            e_grad = nu_* t * th.where(count < c1, -ones , th.where(count > c2, ones , zeros ))
        else:
            # Compute the gradient of the varpi bonus term
            II = th.ones(A.shape).to(device)
            varpi_grad = 2 * beta * mu_ * t * ((II - A) * V_bar ** (beta-1)) @ S

        # Global gradient of the loss with respect to S
        if(isECP):
            grad_loss_S = gamma + kappa_grad + e_grad
        else:
            grad_loss_S = gamma + kappa_grad - varpi_grad


        # Probability matrix evaluation
        P = th.softmax(W.data, dim=2)

        # Global gradient with respect to W
        grad_loss_W = P * (grad_loss_S - th.unsqueeze(th.sum(P * grad_loss_S, 2), 2))
        # equivalent but faster than grad_loss_W = P * (grad_loss_S - (P * grad_loss_S) @ J)

        # First order gradient descent
        W.data -= eta_ * grad_loss_W.data

        # Weight smoothing
        if t % nb_iter == 0 and t > 1 and rho > 1:
            W.data = W.data / rho

        # Store best solution
        if (best_score == 0):

            loss_np = loss.detach().cpu().numpy()
            S_np = S.detach().cpu().numpy()
            solution = S_np[np.argmin(loss_np)]
            end = time.time()
            time_elapsed = end - start
            np.savetxt("Solutions/" + nameGraph + "_solution_k" + str(k) + "_ecp_" + str(isECP) + "_iter_" + str(t) + "_timeEllapsed_" + str(time_elapsed) + "_idx_" + str(idx) + ".csv", solution)
            break;

    # Free gpu to launch another job
    List_free_device.add(gpu)
    print("Gpu released")

    return best_score, t, time_elapsed


# nb of gpu available devices
ngpus = 1

# nb of replications
nbrun = 10

List_free_device = set()
for i in range(ngpus):
    List_free_device.add(i)

# nb of parallel jobs, set default equal to the number of available gpu devices
njobs = ngpus

verbose = True

instanceList=["DSJC250_5-28-GCP","DSJC250_5-29-ECP","DSJC500_5-48-GCP","DSJC500_5-51-ECP","DSJC1000_5-84-GCP","DSJC1000_5-92-ECP"]


#Parameters
D = 200
alpha = 2.5
beta = 1.2
lambda_ = 0.00001
mu_ = 0.000001
nu_ = 0.00001
eta_ = 0.001
sigma_0 = 0.01
nb_iter = 5
rho = 200
max_iter = 2000000

# Launch all instances with nbrun replications in parallel on the available gpu devices with the given parameters
results = Parallel(n_jobs=njobs, backend='threading')(delayed(job_function)( instanceList, D, alpha, beta, lambda_, mu_, nu_, eta_, sigma_0, nb_iter, rho, max_iter, idx=idx) for idx in range(nbrun*len(instanceList)))

#Create result report
freport = pd.DataFrame()

score = []
epoch = []
listTime = []

success = 0.0
avg_time = 0

for i in range(len(results)):
    score.append(results[i][0])
    epoch.append(results[i][1])
    listTime.append(results[i][2])

    if(results[i][0] == 0):
        success += 1
        avg_time += results[i][2]

freport["score"] = score
freport["epoch"] = epoch
freport["time"] = listTime

date = datetime.datetime.now()

freport.to_csv("Reports/report" + str(date) + ".csv", index=False)




