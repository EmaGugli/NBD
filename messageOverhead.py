import numpy as np
import matplotlib.pylab as plt

N = 20 # number of servers
d = 3

def messageOverheadJSQ(T,N):
    return 2*N

def messageOverheadPOD(T,d):
    return 2*d

def messageOverheadJBT(T,N,d):
    return 1 + (N+2*d)/T

T_vec = np.arange(1,1000,50)
JSQ_vec = [messageOverheadJSQ(T, N) for T in T_vec]
POD_vec = [messageOverheadPOD(T, d) for T in T_vec]
JBT_vec = [messageOverheadJBT(T, N, d) for T in T_vec]
plt.plot(T_vec, JSQ_vec, label = 'JSQ', marker='o', ls='--', ms=4.5)
plt.plot(T_vec, POD_vec, label = 'POD-3', marker='*', ls='--')
plt.plot(T_vec, JBT_vec, label = 'JBT-3', marker='D', ls='--', ms=3.5)
plt.xlabel('T (Time slots)')
plt.ylabel('Number of messages per task')
plt.title('Message overhead')
plt.legend()