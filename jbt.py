import math
import random
import time
from random import sample, random 
from collections import defaultdict
import numpy as np
import matplotlib.pylab as plt

n = 100000 # number of samples
Nse= 20 # number of servers
# for jj in range(1,N + 1)
start_time = time.time()
T0 = 1 # unit time 
Ymed = 10 # mean of Y
q  = 3/5 # probability
ET = T0 + (1-q)*Ymed #mean of T

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

alfa = 1/2

def update_status(thr, idList, queueLengths):
    s1, s2, s3 = [server for server in sample(range(Nse), 3)] # select 3 servers at random          
    thr = min(queueLengths[s1], queueLengths[s1], queueLengths[s1])  # take the shortest queue as new threashold
    idList = [server for server in range(1,21) if queueLengths[i+1] < thr]  # update idList with the indexes of the servers having queue length shorter than the threashold
    return thr, idList

def JBTd(queue, idList):
    if len(idList) == 0:
        # in the case that no queue is below the threashold, choose a random server
        return sample(range(1,Nse+1), 1)[0]
    else:
        # if there are some queue with lenght below the throashold, then choose a server among them
        return sample(idList, 1)[0]

delayJBT = defaultdict() # DICTIONARY TO PUT THE VALUE OF RHO AND THE ACCORDING AVERAGE DELAY
for rho in np.arange(0.8, 1, 0.02): # THE RANGE FUNCTION DOESNT WORK FOR DECIMALS SO I HAD TO USE THAT 
    rho = round(rho,2) # HOWEVER THERE ARE SOME PROBLEMS BECAUSE
    print(rho)   
    EX = rho*Nse*ET # calculate the ex according to rho
    beta = EX/math.gamma(1+(1/alfa)) # beta accordinng to ex
    Xv = []
    for i in range(n):
        Xv.append(max(1,min(100*EX,round(beta*(-math.log(R3[i]))**(1/alfa))))) # simulations of Xv 
    #matrix = [1 for i in range(20)]
    #total waiting time
    wt = 0 # delay
    queue = defaultdict(int) # waiting time at server i
    for i in range(1,21):
        queue[i] = 0
    jc = 0 # job counter
    time = 0 # time counter
    thr = 1 # starting threshold
    idList = [i for i in range(1,21)] # list of server's Ids whose queue is below the threashold (all of them at the beginning since they all start empty)
    queueLengths = [0]*20 # store the length of each queue
    idTasksForQueue = defaultdict(int) # store job id
    for i in range(1,21):
        idTasksForQueue[i] = []
    print(queueLengths)   
    while True:
        if (jc == n ): # or all(value == 0 for value in queue.values())):
            break # when the network is empty 
        
        time += Tv[jc] # add the time that pass when there was not arrivals
        
        # EVery 1000 time units update the threshold and the idList in the dispatcher
        if time % 1000 == 0:
            thr, idList = update_status(thr, idList, queueLengths)
        
        for i in range(1,Nse+1):
            queue[i] -= Tv[jc]
            if len(idTasksForQueue[i]) > 0: # remove from the queue all the tasks that are already compleated
                task = idTasksForQueue[i][0]
                while (time - Xv[task] > 0):
                    idTasksForQueue[i] = idTasksForQueue[i][1:]
                    queueLengths[i-1] -= 1
                    task = idTasksForQueue[0]               
            if (queue[i] < 0):
                queue[i] = 0
                idTasksForQueue[i] = []
                queueLengths[i-1] = 0
        
        sn = JBTd(queue, idList) # sn in range(1,21)
        queueLengths[sn-1] += 1
        idTasksForQueue[sn].append(jc)
        queue[sn] += Xv[jc]
        wt += queue[sn]
        jc += 1
    delayJBT[rho] = wt/jc
    
lists = sorted(delayJBT.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(x, y)
plt.show()