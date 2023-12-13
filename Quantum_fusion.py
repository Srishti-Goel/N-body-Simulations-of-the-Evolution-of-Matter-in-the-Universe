import numpy as np
from Quantum_Tunneling import is_tunneled

mass_of_proton = 1.67e-27
proton_proton_barrier = 1.602e-13 # 1MeV in Joules
proton_proton_energy_released = 1.442 * 1.602e-13

def fuse_2_protons(E):
    energy_released = 0
    # Energy released if the fusion happens: 1.442 MeV
    if is_tunneled(E, mass_of_proton,proton_proton_barrier,1e-12):
        energy_released = proton_proton_energy_released
    return energy_released
total_energy = 0
for i in range(int(1e6)):
    total_energy += fuse_2_protons(1.602e-16)
print(total_energy)