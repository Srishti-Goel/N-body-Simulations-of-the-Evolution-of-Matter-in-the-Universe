'''
    This code allows us to play with certain constants like the relative ratios of the matter and dark-matter particles,
    initial universe size, strength of the gravitational force, the fusion pressure (outward, in stars), star-formation
    density threshold, (etc.), and allows it to evolve in a completely Newton's gravity with no cosmological expansion 
    universe.

    We observe that by playing with these variables, we may prolong the life of the universe, but it always eventually 
    collapses into a blob
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# TODO: Shouldn't have too many particles at the same location
# TODO: When the particle hits the boundary?
# TODO: Make a more optimized way to save the video - FAILED
# TODO: Need to generalize the file-format 
# TODO: Start star-formation also ... eh

# Computational Constants
WORLD_SIZE = [51,51, 51] 
                    # Initial matter and dark-matter distribution
NO_OF_TIME_STEPS = 100 
                    # No. of time-steps

DARK_MATTER_PARTICLES = 150 
                    # Initial number of dark-matter particles (do not decay)
NORMAL_MATTER_PARTICLES = 200
                    # Initial number of normal-matter particles (may fuse to )
TOTAL_PARTICLES = NORMAL_MATTER_PARTICLES + DARK_MATTER_PARTICLES

GRAVITATIONAL_CONST = 1 
                    # Constant of proportionality in Newton's law of Gravity
STAR_FORMATION_THRESHOLD = 10 
                    # If the star-formation starts, outward pressure is proportional to density - STAR_FORMATION_THRESHOLD
FUSION_PRESSURE_CONST = 10 
                    # Outward pressure (inverse-square law)
sim_no = 32         # Simulation number (for the output file name)

# Initializing distributions
normal_matter_pos = np.random.random(size=(NORMAL_MATTER_PARTICLES, 3)) * WORLD_SIZE
dark_matter_pos = np.random.random(size=(DARK_MATTER_PARTICLES, 3)) * WORLD_SIZE

normal_matter_vel = np.zeros((NORMAL_MATTER_PARTICLES, 3))
dark_matter_vel =   np.zeros((DARK_MATTER_PARTICLES, 3))

COM = np.zeros((1,3))

# Calculating number of particles nearby (to start fusion) -- ONLY FOR NORMAL MATTER
def no_of_nearby(normal_matter = normal_matter_pos, i = 0, dist = WORLD_SIZE[0] / 10):
    # Returns no. of particles in the dist radius from the i-th particle in the normal_matter distribution,
    #   the COM of the nearby particles (in this radius)
    #   indices of the nearby particles
    
    # Saving self's position
    self_pos = normal_matter[i, :]

    # Initializing output variables
    nearby = 0
    nearby_indices = []
    nearby_COM = np.zeros((1,3))

    # Iterating through the distribution to find the distances
    for idx in range(normal_matter.shape[0]):
        pos_2 = normal_matter[idx, :]
        if i != idx:
            if np.linalg.norm(pos_2 - self_pos) < dist:
                nearby += 1
                nearby_COM = nearby_COM + pos_2
                nearby_indices.append(idx)
    if nearby != 0:
        nearby_COM = nearby_COM / nearby
    return nearby, nearby_COM, nearby_indices

# Time steps
for time in range(NO_OF_TIME_STEPS):
    # Opening plot to plot stars
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    # Calculating the Center of Mass
    COM = np.sum(normal_matter_pos, axis = 0) + np.sum(dark_matter_pos, axis=0)
    COM = COM / (TOTAL_PARTICLES)
    
    # Adding velocity and random temperature stochastic movements
    dark_matter_pos = dark_matter_pos + dark_matter_vel + (np.random.normal(size=dark_matter_pos.shape) - .5) * 3
    normal_matter_pos = normal_matter_pos + normal_matter_vel + (np.random.normal(size=normal_matter_pos.shape) - .5) * 3

    i = -1
    while 1:
        i += 1
        if i >= NORMAL_MATTER_PARTICLES:
            break
        r =  COM - (1 + 1/(TOTAL_PARTICLES))*normal_matter_pos[i,:]
        force = GRAVITATIONAL_CONST * (TOTAL_PARTICLES - 1) * r / np.linalg.norm(r) ** 3

        if time > 5:
            density, nearby_COM, nearby_indices = no_of_nearby(normal_matter=normal_matter_pos, i = i)
            if density >= STAR_FORMATION_THRESHOLD:
                force -= FUSION_PRESSURE_CONST * (density - STAR_FORMATION_THRESHOLD) * (nearby_COM - (normal_matter_pos[i,:]) ) / np.linalg.norm(nearby_COM - normal_matter_pos[i,:])**3
                prob = 0.001 * (density - STAR_FORMATION_THRESHOLD)
                # print(nearby_COM)
                plt.scatter(nearby_COM[:,0], nearby_COM[:, 1], nearby_COM[:, 2], color = "yellow", marker='x')
                for idx in nearby_indices:
                    kill_count = 0
                    temp = np.random.uniform()
                    if temp < prob and idx -1- kill_count < normal_matter_pos.shape[0]:
                        kill_count += 1
                        normal_matter_pos = np.delete(normal_matter_pos, idx - kill_count, 0)
                        normal_matter_vel = np.delete(normal_matter_vel, idx - kill_count, 0)
                        NORMAL_MATTER_PARTICLES -= 1
                        TOTAL_PARTICLES -= 1
                        if idx < i:
                            i -= 1
        # print(i, np.linalg.norm(r), force)

        ##################################
        # BAD COLLISIONS
        ##################################
        if normal_matter_pos[i, 0] >= WORLD_SIZE[0] or normal_matter_pos[i, 1] >= WORLD_SIZE[1] or normal_matter_pos[i, 2] >= WORLD_SIZE[2]:
            normal_matter_vel[i, :] = -1*np.random.random((1,3)) * 5
        elif normal_matter_pos[i, 0] <= 0 or normal_matter_pos[i, 1] <= 0 or normal_matter_pos[i, 2] <= 0:
            normal_matter_vel[i, :] = np.random.random((1,3))* 5
        else:
            normal_matter_vel[i, :] = normal_matter_vel[i, :] + force
        
    for i in range(DARK_MATTER_PARTICLES):
        r =  COM - (1 + 1/(TOTAL_PARTICLES))*dark_matter_pos[i,:]
        force = GRAVITATIONAL_CONST * (TOTAL_PARTICLES - 1) * r / np.linalg.norm(r) ** 3
        # print(i, np.linalg.norm(r), force)

        ##################################
        # BAD COLLISIONS
        ##################################
        if dark_matter_pos[i, 0] >= WORLD_SIZE[0] or dark_matter_pos[i, 1] >= WORLD_SIZE[1] or dark_matter_pos[i, 2] >= WORLD_SIZE[2]:
            dark_matter_vel[i, :] = -1*np.random.random((1,3)) * 2
        elif dark_matter_pos[i, 0] <= 0 or dark_matter_pos[i, 1] <= 0 or dark_matter_pos[i, 2] <= 0:
            dark_matter_vel[i, :] = np.random.random((1,3)) * 2
        else:
            dark_matter_vel[i, :] = dark_matter_vel[i, :] + force
        

    ax.scatter3D(WORLD_SIZE[0], WORLD_SIZE[1], WORLD_SIZE[2])
    ax.scatter3D(normal_matter_pos[:,0], normal_matter_pos[:,1], normal_matter_pos[:,2], color = "green")
    #ax.scatter3D(dark_matter_pos[:,0], dark_matter_pos[:,1], dark_matter_pos[:,2], color = "blue")
    plt.title("Time step: "+str(time + 1))
    
    filename = '3d/3d_vis_'+str(time)+'.png'
    plt.savefig(filename, dpi = 175)

    plt.close()
    if (time + 1) % 10 == 0:
        print("Time:", time+1, "COM:", COM, "No. of normal particles:", NORMAL_MATTER_PARTICLES)
print("Remaining particles\n Normal:", normal_matter_pos.shape[0], "\n Dark:", dark_matter_pos.shape[0])

png_count = NO_OF_TIME_STEPS
files = []
for i in range(png_count):
    seq = str(i)
    file_names = '3d/3d_vis_' + seq + '.png'
    files.append(file_names)
frames = []
for i in files:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save('Newton_simulations_gifs/sim' + str(sim_no) +'.gif', format='GIF', append_images=frames[1:], save_all = True, duration = 100, loop = 0)

#print(normal_matter_pos, dark_matter_pos)