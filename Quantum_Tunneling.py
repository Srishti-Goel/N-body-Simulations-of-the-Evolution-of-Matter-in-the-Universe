import numpy as np
h = 6.626e-34
h_cross = h / (2*np.pi)
def is_tunneled(E, mass, barrier_height, barrier_width):
    prob = np.exp(-2*barrier_width*np.sqrt(np.abs(2*mass*(barrier_height - E)/h_cross**2)))
    # print(prob)

    if np.random.rand() < prob:
        return True
    return False
# count = 0
# for i in range(int(1)):
#     temp = is_tunneled(1e-10, 1e-27, 2e-10, .5e-15)
#     if temp:
#         count += 1
# print(count)