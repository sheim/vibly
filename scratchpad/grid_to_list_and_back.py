import numpy as np


grid_def = (np.linspace(4, 5, 2),
            np.linspace(0, 1, 4))


grid = np.meshgrid(*(grid_def), indexing='ij')

x = np.vstack(map(np.ravel, grid)).T

print(x[:,0].reshape((2,4)))
print(x[:,1].reshape((2,2)))

print('___________________________')

grid_def = (np.linspace(0, 1, 2),
            np.linspace(2, 3, 3),
            np.linspace(4, 5, 4),
            np.linspace(6, 7, 5))

# grid_def = (np.linspace(6, 7, 2),
#             np.linspace(4, 5, 3),
#             np.linspace(2, 3, 4),
#             np.linspace(0, 1, 5))


grid = np.meshgrid(*(grid_def), indexing='ij')

x = np.vstack(map(np.ravel, grid)).T

tmp = list(map(np.ravel, grid))

# x2 = np.array(list(zip(*(x.flat for x in grid))))


print('___')
print(x[:,1].reshape((2,3,4,5)))
print('___')
print(x[:,2].reshape((2,3,4,5)))
print('___')
print(x[:,3].reshape((2,3,4,5)))
print('___')
