import numpy as np
import torch
import math

if __name__ == '__main__':
    a = np.random.rand(2, 2, 3, 2, 2)

    b = a.reshape((-1, 3, 2, 2))

    print(a)
    print('a shape = {}'.format(a.shape))
    print('========================')
    print(b)
    print('b shape = {}'.format(b.shape))
