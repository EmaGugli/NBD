import math
from random import sample, random 
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pylab as plt
import pandas as pd 
import seaborn as sns; sns.set()

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

def myAlgo(queueMyAlgoTime, x):
    if x <= 50:
#        return 1
 #   elif (x > 20 and x <= 50):
        startServers = {key:queueMyAlgoTime[key] for key in range(1,3)}
        minLen = min(startServers.values())
        shortesQueues = [server for server, waitingTime in startServers.items() if waitingTime == minLen]
        return sample(shortesQueues, 1)[0]
    elif (x > 50 and x < 2000):
        midServers = {key:queueMyAlgoTime[key] for key in range(3,20)}
        minLen = min(midServers.values())
        shortesQueues = [server for server, waitingTime in midServers.items() if waitingTime == minLen]
        return sample(shortesQueues, 1)[0]
    else:
        return Nse

def JSQ(queuesJSQ):
    """ select the server with the shortest queue """
    minLen = len(queuesJSQ[min(queuesJSQ, key=lambda x : len(queuesJSQ[x]))])
    shortesQueues = []
    for key, value in queuesJSQ.items():
        if len(value) == minLen:
            shortesQueues.append(key)
    return sample(shortesQueues, 1)[0]

def updateQueues(queue, t, Xv):   
    if len(queue) > 0:
#        print(queue, t)
        for job, value in list(queue.items()):
 #           print(value[1], Xv[value[0]])
            if (value[1] + Xv[value[0]]) <= t:
                queue.pop(job)
    return queue

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
    wtJSQ, wtMyAlgo = 0, 0
    time = 0   
    queuesJSQ = OrderedDict([(i, OrderedDict()) for i in range(1,21)])
    
    queueMyAlgoTime = defaultdict(int)
    for i in range(1,Nse+1):
        queueMyAlgoTime[i] = 0
    
    startVecJSQ = [] # list of starting time of each job
    for jc in range(n):
        x = max(1,min(100*EX,round(beta*(-math.log(R3[jc]))**(1/alfa))))
        Xv.append(x)  
        
        for i in range(1,Nse+1):
            queuesJSQ[i] = updateQueues(queuesJSQ[i], time, Xv)
            
            queueMyAlgoTime[i] -= Tv[jc]
            if (queueMyAlgoTime[i] < 0):
                queueMyAlgoTime[i] = 0
                
        serverJSQ = JSQ(queuesJSQ) # assign the job jc to a server in [1,20]
        startJobJSQ = addJob(serverJSQ, queuesJSQ, time, jc, Xv)       
        startVecJSQ.append(startJobJSQ)
        wtJSQ += startJobJSQ + Xv[jc] - time    
        
        serverMyAlgo = myAlgo(queueMyAlgoTime, x) # return server number
        queueMyAlgoTime[serverMyAlgo] += x
        wtMyAlgo += queueMyAlgoTime[serverMyAlgo]
        
        time += Tv[jc] 
    return wtJSQ, wtMyAlgo, queueMyAlgoTime, Xv

delayJSQ = defaultdict() 
delayMyAlgo = defaultdict() 
rho_vec = np.linspace(0.8,0.99,10)
for rho in rho_vec:
    rho = round(rho,2) 
    wtJSQ, wtMyAlgo, queueMyAlgoTime, Xv = delay(rho)   
    delayJSQ[rho] = wtJSQ/n
    delayMyAlgo[rho] = wtMyAlgo/n
    
listJSQ = sorted(delayJSQ.items()) # sorted by key, return a list of tuples
dfJSQ = pd.DataFrame(listJSQ, columns =['Rho', 'ED']) 
ax = sns.lineplot(x="Rho", y="ED", data=dfJSQ)

listMyAlgo = sorted(delayMyAlgo.items()) # sorted by key, return a list of tuples
df = pd.DataFrame(listMyAlgo, columns =['Rho', 'ED']) 
ax = sns.lineplot(x="Rho", y="ED", data=df)

plt.title('Mean system time comparison for load balancing system algorithms')
plt.xlabel("Load (rho)")
plt.ylabel("Mean system time")
plt.legend()
plt.show()