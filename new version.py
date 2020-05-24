import math
from random import sample, random 
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pylab as plt
from statistics import mean
import pandas as pd

n = 100000 # number of samples
Nse= 20 # number of servers
# for jj in range(1,N + 1)
T0 = 1 # unit time 
Ymed = 10 # mean of Y
q  = 3/5 # probability
ET = T0 + (1-q)*Ymed #mean of T
alfa = 1/2

# generating the random [0,1] numbers for each simulation
R1 = [random() for _ in range(n)]
R2 = [random() for _ in range(n)]
R3 = [random() for _ in range(n)]

Tv = []
for i in range(n):
    if R1[i] <= q :
        Tv.append(T0)
    else:
        Tv.append(round(T0-Ymed*math.log(R2[i])))

def JSQ(queuesJSQ):
    """ select the server with the shortest queue """
    minLen = len(queuesJSQ[min(queuesJSQ, key=lambda x : len(queuesJSQ[x]))])
    shortesQueues = []
    for key, value in queuesJSQ.items():
        if len(value) == minLen:
            shortesQueues.append(key)
    return sample(shortesQueues, 1)[0]

def POD(queuesPOD):
    s = sample(list(queuesPOD.keys()),3)
    sampleServer = {i : queuesPOD[i] for i in s}
    minLen = min(len(queuesPOD[i]) for i in s)
    shortesQueues = []
    for key, value in sampleServer.items():
        if len(value) == minLen:
            shortesQueues.append(key)
    return sample(shortesQueues, 1)[0]

def JBT(idList):
    if len(idList) == 0:
        # in the case that no queue is below the threashold, choose a random server
        return sample(range(1,Nse+1), 1)[0]
    else:
        # if there are some queue with lenght below the throashold, then choose a server among them
        return sample(idList, 1)[0]
    
def updateQueues(queue, t, Xv):   
    if len(queue) > 0:
#        print(queue, t)
        for job, value in list(queue.items()):
 #           print(value[1], Xv[value[0]])
            if (value[1] + Xv[value[0]]) <= t:
                queue.pop(job)
    return queue

def update_status(thr, idList, queuesJBT):
    s = sample(range(1,Nse+1),3)
    sampleServer = {i : queuesJBT[i] for i in s}
    newThr = min(len(sampleServer[i]) for i in s)
    if newThr == 0:
        newThr = 1
    # update idList with the indexes of the servers having queue length shorter than the threashold
    newIdList = [server for server in range(1,21) if len(queuesJBT[server]) < newThr]  
    newMessages = len([id for id in newIdList if id not in idList]) + 3
    return newThr, newIdList, newMessages

def addJob(server, queueDict, time, jc, Xv):
    queue = queueDict[server] #chosen queue
    lenServer = len(queue)
    if lenServer == 0:
        startJob = time
        queue[0] = (jc, startJob)
    else:
        lastJobIndex = list(queue.keys())[-1]
        lastWaitingJob = queue[lastJobIndex] # tuple (jc, startJc)
        startJob = lastWaitingJob[1] + Xv[lastWaitingJob[0]]
        queue[lenServer] = (jc, startJob)
    return startJob

def delay(rho):
    print(rho)
    EX = rho*Nse*ET # calculate E[X] according to rho
    beta = EX/math.gamma(1+(1/alfa)) # beta accordinng to ex
    Xv = []
    wtJSQ, wtPOD, wtJBT = [], [], []
    time = 0   
    queuesPOD = OrderedDict([(i, OrderedDict()) for i in range(1,21)])
    queuesJSQ = OrderedDict([(i, OrderedDict()) for i in range(1,21)])    # dictionary with info about each queue {1: {0: (jc1, startVec[jc]), 1: (jc21, startVec[jc2])}, ...}         
    queuesJBT = OrderedDict([(i, OrderedDict()) for i in range(1,21)])
    startVecJSQ = [] # list of starting time of each job
    startVecPOD = []
    startVecJBT = []
    messJBT = 0
    thr = 1 # starting threshold
    idList = [i for i in range(1,21)]  # list of available servers kept at the dispatcher
    for jc in range(n):
        Xv.append(max(1,min(100*EX,round(beta*(-math.log(R3[jc]))**(1/alfa)))))  
        
        #update the threashold and the list of servers' ids at the dispatcher
        if (jc > 0 and jc % 1000) == 0:
            thr, idList, newMessages = update_status(thr, idList, queuesJBT)
            messJBT += newMessages
        
        for i in range(1,Nse+1):
            queuesJSQ[i] = updateQueues(queuesJSQ[i], time, Xv)
            queuesPOD[i] = updateQueues(queuesPOD[i], time, Xv)
            queuesJBT[i] = updateQueues(queuesJBT[i], time, Xv)
            
        serverJSQ = JSQ(queuesJSQ) # assign the job jc to a server in [1,20]
        startJobJSQ = addJob(serverJSQ, queuesJSQ, time, jc, Xv)       
        startVecJSQ.append(startJobJSQ)
        wtJSQ.append(startJobJSQ + Xv[jc] - time)
        
        serverPOD = POD(queuesPOD) # assign the job jc to a server in [1,20]
        startJobPOD = addJob(serverPOD, queuesPOD, time, jc, Xv)       
        startVecPOD.append(startJobPOD)
        wtPOD.append(startJobPOD + Xv[jc] - time)
        
        serverJBT = JBT(idList) # assign the job jc to a server in [1,20]
        startJobJBT = addJob(serverJBT, queuesJBT, time, jc, Xv)       
        startVecJBT.append(startJobJBT)
        wtJBT.append(startJobJBT + Xv[jc] - time)
        messJBT +=1
        time += Tv[jc] 
    return wtJSQ, wtPOD, wtJBT, messJBT
            
""" Start simulation """
delayJSQ = defaultdict()
delayPOD = defaultdict()
delayJBT = defaultdict() 
messJBTvec = defaultdict() 
rho_vec = [0.8, 0.83, 0.855, 0.88, 0.9,0.92,0.94,0.95,0.96,0.97,0.98,0.985,0.99]
for rho in rho_vec:
    #rho = round(rho,2) 
    wtJSQ, wtPOD, wtJBT, messJBT = delay(rho)    
    delayJSQ[rho] = mean(wtJSQ)
    delayPOD[rho] = mean(wtPOD)
    delayJBT[rho] = mean(wtJBT)
    messJBTvec[rho] = messJBT/n
    
dfJSQ = pd.DataFrame.from_dict(delayJSQ, orient='index', columns =['ED'])
plt.plot(dfJSQ.index, dfJSQ['ED'], marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=2, label = 'JSQ')

dfPOD = pd.DataFrame.from_dict(delayPOD, orient='index', columns =['ED'])
plt.plot(dfPOD.index, dfPOD['ED'], marker='*', markerfacecolor='forestgreen', markersize=8, color='limegreen', linewidth=2, label='POD-3')

dfJBT = pd.DataFrame.from_dict(delayJBT, orient='index', columns =['ED'])
plt.plot(dfJBT.index, dfJBT['ED'], label = "JBT-3", marker='D', markerfacecolor='orangered', markersize=6, color='coral', linewidth=2)


plt.title('Heavy-traffic delay performance')
plt.xlim(0.8,1)
plt.xticks(np.arange(0.8, 1, step=0.02))
plt.xlabel(r'Load coefficient $\rho$')
plt.ylabel("Mean system time")
plt.legend()
plt.show()
                
""" Message overhead """ 
N = 20
d = 3    
def messageOverheadJSQ(T,N):
    return 2*N

def messageOverheadPOD(T,d):
    return 2*d
def messageOverheadJBT(T,N,d):
    return 1 + (N+2*d)/T

JSQ_vec = [messageOverheadJSQ(T, N) for T in rho_vec]
POD_vec = [messageOverheadPOD(T, d) for T in rho_vec]
            
plt.plot(rho_vec, JSQ_vec, label = 'JSQ', marker='o', ls='--', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=2)
plt.plot(rho_vec, POD_vec, label = 'POD-3', ls='--', marker='*', markerfacecolor='forestgreen',markersize=10, color='limegreen', linewidth=2)
plt.plot(rho_vec, list(messJBTvec.values()), label = 'JBT-3', ls='--', marker='D', markerfacecolor='orangered', markersize=6, color='coral', linewidth=2)
plt.xlim(0.8,1)
plt.xticks(np.arange(0.8, 1, step=0.02))
plt.xlabel(r'Load coefficient $\rho$')
plt.ylabel('Number of messages per task')
plt.title('Heavy-traffic message overhead')
plt.legend()            

T_vec = list(np.arange(1,100,33))+list(np.arange(100,1001,50))
JSQ_vec_T = [messageOverheadJSQ(T, N) for T in T_vec]
POD_vec_T = [messageOverheadPOD(T, d) for T in T_vec]
JBT_vec_T = [messageOverheadJBT(T, N, d) for T in T_vec]
plt.plot(T_vec, JSQ_vec_T, label = 'JSQ', marker='o', ls='--', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=2)
plt.plot(T_vec, POD_vec_T, label = 'POD-3', ls='--', marker='*', markerfacecolor='forestgreen',markersize=10, color='limegreen', linewidth=2)
plt.plot(T_vec, JBT_vec_T, label = 'JBT-3', ls='--', marker='D', markerfacecolor='orangered', markersize=6, color='coral', linewidth=2)
plt.xlim(0,1000)
#plt.xticks(np.arange(0.8, 1, step=0.02))
plt.xlabel("T (Time slots)")
plt.ylabel('Number of messages per task')
plt.title('Message overhead by time slots')
plt.legend()   




JBT_val = list(messJBTvec.values())
dfmessT = pd.DataFrame(list(zip(JSQ_vec_T, POD_vec_T, JBT_vec_T)), columns = ['JSQ', 'POD', 'JBT'], index=T_vec)
dfmessRho = pd.DataFrame(list(zip(JSQ_vec, POD_vec, JBT_val)), columns = ['JSQ', 'POD', 'JBT'], index=rho_vec)

fig, axs = plt.subplots(2, figsize=(6,8))
axs[0].plot(dfmessRho.index, dfmessRho['JSQ'], marker='o', ls='--', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=2, label = 'JSQ')
axs[0].plot(dfmessRho.index, dfmessRho['POD'], label = 'POD-3', ls='--', marker='*', markerfacecolor='forestgreen',markersize=10, color='limegreen', linewidth=2)
axs[0].plot(dfmessRho.index, dfmessRho['JBT'], label = 'JBT-3', ls='--', marker='D', markerfacecolor='orangered', markersize=6, color='coral', linewidth=2)
#axs[0].xlim(0.8,1)
#axs[0].xticks(np.arange(0.8, 1, step=0.02))
axs[0].set(xlabel=r'Load coefficient $\rho$', ylabel='Messages per task')
axs[0].set_title('Heavy-traffic message overhead')
axs[0].legend()  
  
axs[1].plot(dfmessT.index, dfmessT['JSQ'], marker='o', ls='--', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=2, label = 'JSQ')
axs[1].plot(dfmessT.index, dfmessT['POD'], label = 'POD-3', ls='--', marker='*', markerfacecolor='forestgreen',markersize=10, color='limegreen', linewidth=2)
axs[1].plot(dfmessT.index, dfmessT['JBT'], label = 'JBT-3', ls='--', marker='D', markerfacecolor='orangered', markersize=6, color='coral', linewidth=2)
axs[1].set(xlabel='T (Time slots)', ylabel='Messages per task')
axs[1].set_title('Message overhead by time slots')
axs[1].legend() 

plt.plot(dfmessRho.index, dfmessRho['JSQ'], marker='o', ls='--', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=2, label = 'JSQ')
plt.plot(dfmessRho.index, dfmessRho['POD'], label = 'POD-3', ls='--', marker='*', markerfacecolor='forestgreen',markersize=10, color='limegreen', linewidth=2)
plt.plot(dfmessRho.index, dfmessRho['JBT'], label = 'JBT-3', ls='--', marker='D', markerfacecolor='orangered', markersize=6, color='coral', linewidth=2)
plt.xlabel(r'Load coefficient $\rho$')
plt.ylabel('Messages per task')
plt.title('Heavy-traffic message overhead')
plt.legend()  
