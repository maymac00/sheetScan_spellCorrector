from scanning import *

k = gaussian_kernel(3, 2)
mat = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]])
res = np.dot(mat, k)
print(res)
