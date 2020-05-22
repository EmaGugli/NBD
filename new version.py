import math
from random import sample, random 
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pylab as plt

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
    thr = min(len(sampleServer[i]) for i in s)
    if thr == 0:
        thr = 1
    idList = [server for server in range(1,21) if len(queuesJBT[server]) < thr]  # update idList with the indexes of the servers having queue length shorter than the threashold
    return thr, idList

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
    wtJSQ, wtPOD, wtJBT = 0, 0, 0
    time = 0   
    queuesPOD = OrderedDict([(i, OrderedDict()) for i in range(1,21)])
    queuesJSQ = OrderedDict([(i, OrderedDict()) for i in range(1,21)])    # dictionary with info about each queue {1: {0: (jc1, startVec[jc]), 1: (jc21, startVec[jc2])}, ...}         
    queuesJBT = OrderedDict([(i, OrderedDict()) for i in range(1,21)])
    startVecJSQ = [] # list of starting time of each job
    startVecPOD = []
    startVecJBT = []
    thr = 1 # starting threshold
    idList = [i for i in range(1,21)]
    for jc in range(n):
        Xv.append(max(1,min(100*EX,round(beta*(-math.log(R3[jc]))**(1/alfa)))))  
        
        if (jc > 0 and jc % 1000) == 0:
            thr, idList = update_status(thr, idList, queuesJBT)
        
        for i in range(1,Nse+1):
            queuesJSQ[i] = updateQueues(queuesJSQ[i], time, Xv)
            queuesPOD[i] = updateQueues(queuesPOD[i], time, Xv)
            queuesJBT[i] = updateQueues(queuesJBT[i], time, Xv)
            
        serverJSQ = JSQ(queuesJSQ) # assign the job jc to a server in [1,20]
        startJobJSQ = addJob(serverJSQ, queuesJSQ, time, jc, Xv)       
        startVecJSQ.append(startJobJSQ)
        wtJSQ += startJobJSQ + Xv[jc] - time
        
        serverPOD = POD(queuesPOD) # assign the job jc to a server in [1,20]
        startJobPOD = addJob(serverPOD, queuesPOD, time, jc, Xv)       
        startVecPOD.append(startJobPOD)
        wtPOD += startJobPOD + Xv[jc] - time
        
        serverJBT = JBT(idList) # assign the job jc to a server in [1,20]
        startJobJBT = addJob(serverJBT, queuesJBT, time, jc, Xv)       
        startVecJBT.append(startJobJBT)
        wtJBT += startJobJBT + Xv[jc] - time
        
        time += Tv[jc] 
    return wtJSQ, wtPOD, wtJBT
            
""" Start simulation """
delayJSQ = defaultdict()
delayPOD = defaultdict()
delayJBT = defaultdict() 
rho_vec = np.linspace(0.8,0.99,10)
for rho in rho_vec:
    rho = round(rho,2) 
    wtJSQ, wtPOD, wtJBT = delay(rho)    
    delayJSQ[rho] = wtJSQ/n
    delayPOD[rho] = wtPOD/n
    delayJBT[rho] = wtJBT/n
    
listJSQ = sorted(delayJSQ.items()) # sorted by key, return a list of tuples
xJSQ, yJSQ = zip(*listJSQ) # unpack a list of pairs into two tuples
plt.plot(xJSQ, yJSQ, label = "JSQ")

listPOD = sorted(delayPOD.items()) # sorted by key, return a list of tuples
xPOD, yPOD = zip(*listPOD) # unpack a list of pairs into two tuples
plt.plot(xPOD, yPOD, label = "POD-3")

listJBT = sorted(delayJBT.items()) # sorted by key, return a list of tuples
xJBT, yJBT = zip(*listJBT) # unpack a list of pairs into two tuples
plt.plot(xJBT, yJBT, label = "JBT-3")

plt.title('Mean system time comparison for load balancing system algorithms')
plt.xlabel("Load (rho)")
plt.ylabel("Mean system time")
plt.legend()
plt.show()
                
            
            
            