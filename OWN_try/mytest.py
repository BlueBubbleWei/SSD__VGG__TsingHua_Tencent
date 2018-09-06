import numpy as np
arr=np.array([[1,23,45],[23,12,4]])
# print(np.max(arr))
arr=[[3,4],[3,4],[5,6]]
for i,name in zip(range(len(arr)),arr):
    print(i,name)

for i,name in enumerate(arr):
    print(i,name)