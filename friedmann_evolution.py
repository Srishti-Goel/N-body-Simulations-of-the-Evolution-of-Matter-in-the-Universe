''' 
---------------LET'S REVIEW THE REQUIRED THEORY!---------
    The Friedmann Equations:
        a''/a = -4piG/3(rho + 3p/c^2) + Lambda c^2 / 3
        (a'^2 + kc^2) / a^2 = (8 pi G rho + Lambda c^2) / 3
    Hubble's constant : Age propto 1/ H
    H = (a'/a)
    velocity of recession = H * Physical distance

    Physical distance = a(t) * Coordinate distance

    Cosmological Red-shift:
        a(time_observed) / a(time_sourced) = 1 + Z
    k = -2 Energy of the entire universe / c^2

    Curvature of the universe = omega = rho / rho_critical (rho_critical = 3H^2 / 8 pi G)
    a = A t^(2/3) in the critical case
    Phsical distance of the horizon = 3c*t = 2cH

    In a closed universe, time of the univ = 2 pi alpha / c
        where alpha = 4 pi G rho (a / sqrt(k))^3 / 3 c^2
    
    Friedmann-Lemaitre-Robertson-Walker metric :
        ds^2 = -cdt^2 + a^2(t) [dr^2/(1 - kr^2) + r^2 (dtheta^2 + sin^2 theta dphi^2)]
    Gedesic equation: 
        d^2 x^i / ds^2 = -GAMMA ^i _{jk} dx^j/ds dx^k/ds
        Where the Christoffel symbol GAMMA^i_{jk} = 0.5g^{il}[dg_{lk}/dx^j + dg_{lj} / dx^k - d g_{jk} / dx^l]
    g^{ij} = g_{ij}^-1
'''

# Would it be easier to save each individual "particle" / small element of dust separately?
# This way we may save the position, the velocity of each "particle" and compute.
# But what if 2 "particles" have the same position and different velocities?
#       Pressure and energy can add up. So I guess we can still treat them as different particles,
#       experiencing the same a(t), r, theta and phi


# ======IMPORTING-----
import numpy as np

# ------IMPORTANT CONSTANTS FOR COMPUTATION------
ALL_DIMENSIONS = [0,1,2,3]
NO_OF_DIMENSIONS = len(ALL_DIMENSIONS)
SPATIAL_DIMENSIONS = [1,2,3]

# CONSERVATION OF ENERGY( of each "particle" ):

# Let's say we have T^i_j and gamma^i_{jk}, d_a(T^b_c)
# Conservation of energy??? Not if the particles may leave the box, na? But they can't. Because the box is defined to contain them, though it can move itself
# LHS = Delta_a T^a_0

# dT[a,b,c] = d/dx^a (T^b_c)
# gamma[a,b,c] = gamma^a_{bc}
# T[a,b] = T^a_b
for i in range(10):
    # Random initializations:
    dT = np.zeros((NO_OF_DIMENSIONS, NO_OF_DIMENSIONS, NO_OF_DIMENSIONS))
    # print('dT:', dT)
    gamma = np.random.uniform(size=(NO_OF_DIMENSIONS, NO_OF_DIMENSIONS, NO_OF_DIMENSIONS))
    # print("Gamma:", gamma)
    T = np.random.uniform(size=(NO_OF_DIMENSIONS, NO_OF_DIMENSIONS))
    # print("T:", T)

    LHS = 0
    for i in ALL_DIMENSIONS:
        LHS += dT[i, i,0]
        for j in ALL_DIMENSIONS:
            LHS += gamma[i,i,j]*T[j,0] - gamma[i,j,0]*T[j,i]
    if abs(LHS) < 0.5:
        print("LHS:", LHS)