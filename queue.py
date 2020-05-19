# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:02:23 2020

@author: Clara
"""
from random import random
import queueing_tool as qt
import numpy as np
from math import log, modf, sqrt, gamma
from statistics import mean, stdev
from matplotlib import pyplot as plt

q = 3/5
T0 = 1
EY = 10
ET = T0 + (1-q)*EY
ro_vec = np.linspace(0.8,0.99,100)
alpha = 1/2
N = 20

""" Graph setup"""
adja_list = {0: [1], 1: [k for k in range(2, 22)]}
edge_list = {0: {1: 1}, 1: {k: 2 for k in range(2, 22)}}
g = qt.adjacency2graph(adjacency=adja_list, edge_type=edge_list)

def arr_f(t):
    """ return the time of next task arrival"""
    r1, r2 = random(), random()
    if r1 < q: return t + modf(T0 - EY*log(r2))[1]
    else: return t + T0

#def serviceValues(ro):
#    beta = ro*50
#    EX = beta*2 # ro*N*ET
#    return ro, beta, EX
def serviceValues(ro):
    """ evaluate the values of E[X] and beta for the given ro"""
    EX = ro*N*ET
    beta = EX/2
    return ro, beta, EX

def ser_f(t):   
    """ return the end time of the task starting at t"""
    return t + max(1, min(100*EX, modf(beta*(-log(random()))**(1/alpha))[1]))


""" Simulation """
system_delay = []
nSim = 10000
for ro in ro_vec:
    beta, EX = serviceValues(ro)[1:]
    """ Network setup"""
    q_classes = {1: qt.QueueServer, 2: qt.QueueServer}  # class 1 describes the queue at the load balancer, class 2 that of any server
    q_args = {
            1: {
                'arrival_f': arr_f,
                'service_f': lambda t: t,
                'AgentFactory': qt.GreedyAgent # this choose the shortest queue, i.e. JSQ
                },
            2: {
                'num_servers': 1,
                'service_f': ser_f
                }
            }
    qn = qt.QueueNetwork(g=g, q_classes=q_classes, q_args=q_args)
    
    """ Simulation """
    qn.initialize(edge_type=1)
    qn.start_collecting_data()
    qn.simulate(n = nSim)
   
    """ Collect results from the simulation """ 
    data = qn.get_agent_data(edge_type=[1,2])
    delay_vec = []
    for key, value in data.items():
        if len(value) > 1:
            delay = value[1][2] - value[0][0]  # departure from the server - arrival at the load balancer
            if delay > 0 : delay_vec.append(delay)
    system_delay.append(mean(delay_vec))
    
    qn.clear_data()

plt.plot(ro_vec, system_delay)        
plt.title('JSQ algorithm')
plt.xlabel('ro')
plt.ylabel('system delay')

#mymodel = np.poly1d(np.polyfit(ro_vec, system_delay, 3))
#plt.scatter(ro_vec, system_delay)
#plt.plot(ro_vec, mymodel(ro_vec))
#plt.show()

# Visualization
#q_classes = {1: qt.QueueServer, 2: qt.QueueServer}
#q_args = {
#    1: {
#        'arrival_f': arr_f,
#        'service_f': lambda t: t,
#        'AgentFactory': qt.GreedyAgent
#    },
#    2: {
#        'num_servers': 1,
#        'service_f': ser_f
#    }
#}
#qn = qt.QueueNetwork(g=g, q_classes=q_classes, q_args=q_args, seed=13)
#qn.g.new_vertex_property('pos')
#pos = {}
#for v in qn.g.nodes():
#    if v == 0:
#        pos[v] = [0, 0.8]
#    elif v == 1:
#        pos[v] = [0, 0.4]
#    else:
#        pos[v] = [-5. + (v - 2.0) / 2, 0]
#qn.g.set_pos(pos)

#qn.initialize(edge_type=1)
#qn.start_collecting_data()
#qn.simulate(10000)
#data = qn.get_queue_data()
#
#arrivalTime = [data[i][0] for i in range(10000)]  # first column : arrival time
#departureTime = [data[i][2] for i in range(13222)] # third column : departure time
#delay_vec = [departureTime[i] - arrivalTime[i] for i in range(13222)]
#
#qn.clear_data()