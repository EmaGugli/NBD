import math
from random import sample, random 
from collections import defaultdict
import numpy as np
import matplotlib.pylab as plt

n = 10000 # number of samples
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

def JSQ(queueLengthsJSQ):
    """ select the server with the shortest queue """
    shortesQueues = [i+1 for i, x in enumerate(queueLengthsJSQ) if x == min(queueLengthsJSQ)]
    return sample(shortesQueues, 1)[0]

def POD(queueLengthsPOD):
    """ randomly sample the queue lengths of 3 servers and join the shortest among them """
    s = sample(range(20),3) # 0-19
    sampleLengths = {index: queueLengthsPOD[index] for index in s}
    m = min(sampleLengths.items(), key=lambda x: x[1])
    shortestQueues = list()
    for key, value in sampleLengths.items():
        if value == m[1]:
            shortestQueues.append(key)
    sn = sample(shortestQueues, 1)[0] + 1
    return sn

def update_status(thr, idList, queueLengths):
    s1, s2, s3 = [server for server in sample(range(Nse), 3)] # select 3 servers at random          
    thr = min(queueLengths[s1], queueLengths[s1], queueLengths[s1])  # take the shortest queue as new threashold
    if thr == 0:
        thr = 1
    idList = [server for server in range(1,21) if queueLengths[server-1] < thr]  # update idList with the indexes of the servers having queue length shorter than the threashold
    return thr, idList

def JBTd(idList):
    if len(idList) == 0:
        # in the case that no queue is below the threashold, choose a random server
        return sample(range(1,Nse+1), 1)[0]
    else:
        # if there are some queue with lenght below the throashold, then choose a server among them
        return sample(idList, 1)[0]


delayJSQ = defaultdict()
delayPOD = defaultdict()
delayJBT = defaultdict() # DICTIONARY TO PUT THE VALUE OF RHO AND THE ACCORDING AVERAGE DELAY
rho_vec = np.linspace(0.8,0.99,10)
for rho in rho_vec:
    rho = round(rho,2) 
    print(rho)   
    EX = rho*Nse*ET # calculate E[X] according to rho
    beta = EX/math.gamma(1+(1/alfa)) # beta accordinng to ex
    Xv = []
    wtJSQ, wtPOD, wtJBT = 0, 0, 0 # delay
    queueJSQ = defaultdict(int) # waiting time at server i
    queuePOD = defaultdict(int)
    queueJBT = defaultdict(int)
    for i in range(1,21):
        queueJSQ[i] = 0
        queuePOD[i] = 0
        queueJBT[i] = 0
    time = 0 # time counter
      
    queueLengthsJSQ = [0]*20
    idTasksForQueueJSQ = defaultdict(int)
    for i in range(1,21):
        idTasksForQueueJSQ[i] = []    
    
    queueLengthsPOD = [0]*20
    idTasksForQueuePOD = defaultdict(int)
    for i in range(1,21):
        idTasksForQueuePOD[i] = []
    
    idTasksForQueueJBT = defaultdict(int) # store job id 
    queueLengthsJBT = [0]*20 # store the length of each queue    
    for i in range(1,21):
        idTasksForQueueJBT[i] = []
    thr = 1 # starting threshold
    idList = [i for i in range(1,21)] # list of server's Ids whose queue is below the threashold (all of them at the beginning since they all start empty)
          
    """ Start simulating for the given rho """
    for jc in range(n):
        Xv.append(max(1,min(100*EX,round(beta*(-math.log(R3[jc]))**(1/alfa))))) # simulations of Xv         
        time += Tv[jc] # add the time that pass when there was not arrivals
        
        # JBT: Every 100 events update the threshold and the idList in the dispatcher
        if jc % 1000 == 0:
            thr, idList = update_status(thr, idList, queueLengthsJBT)
        
        # update the queues
        for i in range(1,Nse+1):
            queueJSQ[i] -= Tv[jc]
            queuePOD[i] -= Tv[jc]
            queueJBT[i] -= Tv[jc]
            if len(idTasksForQueueJBT[i]) > 0: # remove from the queue all the tasks that are already compleated
                task = idTasksForQueueJBT[i][0]
                while (Tv[jc] - Xv[task] > 0):
                    idTasksForQueueJBT[i] = idTasksForQueueJBT[i][1:]
                    queueLengthsJBT[i-1] -= 1
                    task = idTasksForQueueJBT[0]
            if len(idTasksForQueueJSQ[i]) > 0: # remove from the queue all the tasks that are already compleated
                task = idTasksForQueueJSQ[i][0]
                while (Tv[jc] - Xv[task] > 0):
                    idTasksForQueueJSQ[i] = idTasksForQueueJSQ[i][1:]
                    queueLengthsJSQ[i-1] -= 1
                    task = idTasksForQueueJSQ[0] 
            if (queueJBT[i] < 0):
                queueJBT[i] = 0
                idTasksForQueueJBT[i] = []
                queueLengthsJBT[i-1] = 0
            if (queueJSQ[i] <0):
                queueJSQ[i] = 0
                idTasksForQueueJSQ[i] = []
                queueLengthsJSQ[i-1] = 0
            if (queuePOD[i] <0):
                queuePOD[i] = 0
                idTasksForQueuePOD[i] = []
                queueLengthsPOD[i-1] = 0

        snJSQ = JSQ(queueLengthsJSQ)
        queueLengthsJSQ[snJSQ-1] += 1
        idTasksForQueueJSQ[snJSQ].append(jc)
        queueJSQ[snJSQ] += Xv[jc]
        wtJSQ += queueJSQ[snJSQ]
        
        snPOD = POD(queueLengthsPOD)
        queueLengthsPOD[snPOD-1] += 1
        idTasksForQueuePOD[snPOD].append(jc)
        queuePOD[snPOD] += Xv[jc]
        wtPOD += queuePOD[snPOD]
        
        snJBT = JBTd(idList) # sn in range(1,21)
        queueLengthsJBT[snJBT-1] += 1
        idTasksForQueueJBT[snJBT].append(jc)
        queueJBT[snJBT] += Xv[jc]        
        wtJBT += queueJBT[snJBT]
        
        jc += 1
    
    delayJSQ[rho] = wtJSQ/jc
    delayPOD[rho] = wtPOD/jc
    delayJBT[rho] = wtJBT/jc
    
listJBT = sorted(delayJBT.items()) # sorted by key, return a list of tuples
xJBT, yJBT = zip(*listJBT) # unpack a list of pairs into two tuples
plt.plot(xJBT, yJBT, label = "JBT-3")

listJSQ = sorted(delayJSQ.items()) # sorted by key, return a list of tuples
xJSQ, yJSQ = zip(*listJSQ) # unpack a list of pairs into two tuples
plt.plot(xJSQ, yJSQ, label = "JSQ")

listPOD = sorted(delayPOD.items()) # sorted by key, return a list of tuples
xPOD, yPOD = zip(*listPOD) # unpack a list of pairs into two tuples
plt.plot(xPOD, yPOD, label = "POD")
plt.xlabel("rho")
plt.ylabel("mean system time")
plt.legend()
plt.show()