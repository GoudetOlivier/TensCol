import numpy as np

def computeLossWithLoop(A,S):

    cpt = 0

    for v in range(A.shape[0]):

        colorV = S[v]

        for i in range(A.shape[1]):

            if(A[v,i]==1):

                colorVprim =  S[i]

                if(colorVprim == colorV):

                    cpt+=1

    return cpt/2

S = np.loadtxt("Solutions/flat1000_76_ECP_solution_k91.csv")

k = int(np.max(S)) + 1

print("nb of colors used")
print(k)

filename = "benchmark/flat1000_76.csv"


graph = np.loadtxt(filename, delimiter = ",")
size= int(np.max(graph))

A = np.zeros((size,size))

for i in range(graph.shape[0]):
    A[int(graph[i][0])-1,int(graph[i][1])-1] = 1


A = A + np.transpose(A)


score = computeLossWithLoop(A,S)

print("nb conflicts " + str(score))


allGroups = np.zeros(k)

for i in range(S.shape[0]):
    allGroups[int(S[i])] += 1

print("size groups")
print(allGroups)
