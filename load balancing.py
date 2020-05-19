# -*- coding: utf-8 -*-
"""
Created on Tue May 12 23:21:48 2020

@author: Clara
"""
from random import random
from math import log, modf, sqrt, gamma
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean, stdev

n = 100000 # Simulation size

"""Inter-arrival times"""
T0 = 1
EY = 10
q = 3/5
ET = T0 + (1-q)*EY
R1 = [random() for _ in range(n)]
R2 = [random() for _ in range(n)]
R3 = [random() for _ in range(n)]
T = [1 for _ in range(n)]
for i in range(n):
    if R1[i] > q: T[i] = modf(T0 - EY*log(R2[i]))[1]
ET_sim, sdT_sim = mean(T), stdev(T)        

#Histogram
bins = np.arange(0, 150, 3)
plt.xlim([0, max(T)])
plt.hist(T, bins=bins)
plt.title('Frequency distribution of inter-arrival times')
plt.xlabel('variable T (bin size = 5)')
plt.ylabel('frequency')
plt.show()

"""Service times"""
alpha = 1/2
N = 20 # number of servers
def beta(ro, N, alpha, ET): # ro is the only unknown!
    return ro*N*ET/gamma(1+1/alpha)
def EX(beta):
    return beta*gamma(1+1/alpha)
ro_vec = np.linspace(0.8,0.99,100)
plt.plot(ro_vec,beta(ro_vec, N, alpha, ET), 'b')
plt.xlabel('ro')
plt.ylabel('beta')

beta_vec = beta(ro_vec, N, alpha, ET)
EX_vec = EX(beta_vec)
plt.plot(beta_vec, EX_vec,'r')
plt.xlabel('beta')
plt.ylabel('E[X]')

def serviceValues(ro):
    beta = ro*50
    EX = beta*2
    return ro, beta, EX
plt.plot(ro_vec,serviceValues(ro_vec)[2], 'r')

# Simulation for maximum utilization coefficient
EX_max = serviceValues(0.99)[2]
beta_max = serviceValues(0.99)[1]
X = [1 for _ in range(n)]
for i in range(n):
    X[i] = max(1, min(100*EX_max, modf(beta_max*(-log(R3[i]))**(1/alpha))[1]))
EX_sim, sdX_sim = mean(X), stdev(X)

delay_vec_ro = []
# Simulation of the load balancing system: JSQ
for ro in ro_vec:
    print(ro)
    EX = serviceValues(ro)[2]
    beta = serviceValues(ro)[1]
    q_vec = [[i, 0] for i in range(1,N+1)] # vector reporting the lenght of the queue at each server
    X = [1 for _ in range(n)]
    t = 0
    delay_vec = []
    for i in range(n):
        t += T[i]       
        X[i] = max(1, min(100*EX, modf(beta*(-log(R3[i]))**(1/alpha))[1])) # evaluate the service time for the given task       
        # Unqueue all the tasks that are finished by the actual time t
        for queue in q_vec:
            updated_queue = list(queue[:2])
            for task in range(2,queue[1]+2):
                if t < queue[task][2]:
                    updated_queue.append(queue[task:])
                    break
            updated_queue[1] = len(updated_queue)-2
            queue = updated_queue
        # after this step all the queues contains only the tasks that are executing and that are waiting to be served    
        waiting_time = 0
        # always assign the task to the first server since the queue is ordered
        if q_vec[0][1] == 0: # empty queue
            waiting_time += X[i]
            q_vec[0][1] = 1
            q_vec[0].append((i, X[i], t)) # (task i, service time of task i, starting time)
        else:
            start = q_vec[0][-1][2] +  q_vec[0][-1][1]                
            q_vec[0].append((i, X[i], start)) # the new task can already be served
            q_vec[0][1] += 1
            waiting_time = start - t
            
        delay_vec.append(T[i] + waiting_time + X[i])
        # the task goes to server 1 -> update the queue vector keeping the order
        q_vec.sort(key=lambda d: d[1])
    
    delay_vec_ro.append(mean(delay_vec))
            
plt.plot(ro_vec, delay_vec_ro)        