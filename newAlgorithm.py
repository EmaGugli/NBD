import math
from random import sample, random 
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pylab as plt
import pandas as pd 
import seaborn as sns; sns.set()
import scipy
import statistics
n = 100000 # number of samples
Nse= 20 # number of servers
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
    wtJSQ, wtMyAlgo = [], []
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
        wtJSQ.append(startJobJSQ + Xv[jc] - time)    
        
        serverMyAlgo = myAlgo(queueMyAlgoTime, x) # return server number
        queueMyAlgoTime[serverMyAlgo] += x
        wtMyAlgo.append(queueMyAlgoTime[serverMyAlgo])
        
        time += Tv[jc] 
    return wtJSQ, wtMyAlgo, queueMyAlgoTime, Xv

delayJSQ, delayMyAlgo = defaultdict(), defaultdict() 
rho_vec = np.linspace(0.8,0.99,10)
for rho in rho_vec:
    rho = round(rho,2) 
    wtJSQ, wtMyAlgo, queueMyAlgoTime, Xv = delay(rho)   
    delayJSQ[rho] = (np.mean(wtJSQ), scipy.stats.sem(wtJSQ)*scipy.stats.t.ppf((1 + 0.95) / 2., n-1)) 
    delayMyAlgo[rho] = (np.mean(wtMyAlgo), scipy.stats.sem(wtMyAlgo)* scipy.stats.t.ppf((1 + 0.95) / 2., n-1))
    
dfJSQ = pd.DataFrame.from_dict(delayJSQ, orient='index', columns =['ED', 'ERR'])
plt.plot(dfJSQ.index, dfJSQ['ED'], marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=2, label = 'JSQ')
#ax = sns.lineplot(x="Rho", y="ED", data=dfJSQ)
#plt.errorbar(dfJSQ['Rho'], dfJSQ['ED'], yerr=dfJSQ['ERR'], fmt='o', color='black',
#             ecolor='lightgray', elinewidth=3, capsize=0);

df = pd.DataFrame.from_dict(delayMyAlgo, orient='index', columns =['ED', 'ERR'])
plt.plot(df.index, df['ED'], marker='*', markerfacecolor='red', markersize=10, color='orange', linewidth=2, label = 'New policy')
#plt.errorbar(df['Rho'], df['ED'], yerr=df['ERR'], fmt='o', color='red',ecolor='lightgray', elinewidth=3, capsize=0);

plt.title('Mean system time comparison: \n JSQ vs our new policy')
plt.xlabel(r'Load coefficient $\rho$')
plt.ylabel("Mean system time")
plt.legend()
plt.show()
