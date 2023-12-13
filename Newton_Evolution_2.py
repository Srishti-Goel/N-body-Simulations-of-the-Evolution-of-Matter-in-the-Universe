import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import datetime
import os

INITIAL_WORLD_SIZE = [50, 50, 10]
SIMULATION_TIMES = 5000
INITIAL_MEAN_LENGTH = 0
NO_OF_NORMAL_PARTICLES = 300
NO_OF_DARK_PARTICLES = 2100
BIG_G = 6.67e-1
SIM_NO = 1

while 1:
    file_name = 'Newton_simulations_gifs_v2/sim' + str(SIM_NO) + '.mp4'
    if not os.path.exists(file_name):
        break
    SIM_NO += 1

WIDTH = 1920
HEIGHT = 1080
FPS = 10

forcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Newton_simulations_gifs_v2/sim' +
                      str(SIM_NO) + '.mp4', forcc, FPS, (WIDTH, HEIGHT))

def multiple_gravitational_acceleration(particle_pos, mass_of_particles, pos):
    no_particles = particle_pos.shape[0]
    force = np.zeros(pos.shape)

    # print(pos.shape)

    for i in range(no_particles):
        temp = particle_pos[i, :]
        r = pos - temp
        # print("r:", r)
        dist = np.linalg.norm(r, axis=1)**3
        dist = dist[:, np.newaxis]
        # print("dist:", dist)

        # print(r.shape, force.shape, dist)
        # print(r / dist)
        force += np.nan_to_num(r / dist)*mass_of_particles[i]
    # print("force:", force)
    return -1 * force * BIG_G


def individual_acceleration(normal_pos, dark_pos, pos):
    no_normal = normal_pos.shape[0]
    no_dark = dark_pos.shape[0]
    # print(pos.shape, no_normal)
    # print("Pos:\n", pos, "\nNormal pos:\n",normal_pos)
    force = np.zeros(pos.shape)
    # print(force,pos)
    print(len(pos.shape))

    for i in range(no_normal):
        temp = normal_pos[i, :]
        # print(temp.shape)
        if np.any(temp == pos):
            # print("Found common")
            # print(temp)
            continue
        r = temp - pos
        dist = np.linalg.norm(r)
        force += r / dist**3
        # print("Force:", force, "r:", r)
    for i in range(no_dark):
        temp = dark_pos[i, :]
        # print(temp.shape)
        if np.any(temp == pos):
            # print("Found common")
            # print(temp)
            continue
        r = temp - pos
        dist = np.linalg.norm(r)
        force += r / dist**3
    return force

normal_pos = np.random.randn(NO_OF_NORMAL_PARTICLES, 3) * INITIAL_WORLD_SIZE
dark_pos = np.random.randn(NO_OF_DARK_PARTICLES, 3) * INITIAL_WORLD_SIZE

normal_vel = np.random.rand(NO_OF_NORMAL_PARTICLES, 3) * INITIAL_MEAN_LENGTH
dark_vel = np.random.rand(NO_OF_DARK_PARTICLES, 3) * INITIAL_MEAN_LENGTH

# print(normal_pos.shape, normal_pos[1:5, :])

tic = time.time()

for i in range(SIMULATION_TIMES):
    all_pos = np.concatenate([normal_pos, dark_pos])
    normal_vel += multiple_gravitational_acceleration(all_pos, np.ones(
        shape=(NO_OF_NORMAL_PARTICLES+NO_OF_DARK_PARTICLES, 1)), normal_pos)
    dark_vel += multiple_gravitational_acceleration(all_pos, np.ones(
        shape=(NO_OF_NORMAL_PARTICLES+NO_OF_DARK_PARTICLES, 1)),  dark_pos)

    dark_pos += dark_vel
    normal_pos += normal_vel

    normal_pos -= np.mean(all_pos) - 5000
    dark_pos -= np.mean(all_pos) - 5000

    WORLD_SIZE = np.max(normal_pos, axis=0)
    print(f'\r{(i+1)*1000//SIMULATION_TIMES/10}%\t|\tTime remaining: {datetime.timedelta(seconds=(SIMULATION_TIMES-i)//((i+1)/(time.time() - tic + 1)))}', end='')

    # fig = plt.figure(figsize = (10, 7))
    # ax = plt.axes(projection ="3d")
    # ax.scatter3D(normal_pos[:,0], normal_pos[:,1], normal_pos[:,2])
    # ax.scatter3D(normal_pos[:,0]/WORLD_SIZE[0], normal_pos[:,1]/WORLD_SIZE[1], normal_pos[:,2]/WORLD_SIZE[2])
    # plt.title("Time step: "+str(i + 1))

    # filename = '3d_v2/3d_vis_'+str(i)+'.png'
    # plt.savefig(filename, dpi = 75)
    # plt.close()
    frame = np.array(np.zeros((HEIGHT, WIDTH, 3)), dtype=np.uint8)
    max_x = min(max(all_pos[:, 0]), 10000)
    max_y = min(max(all_pos[:, 1]), 10000)
    max_z = min(max(dark_pos[:, 2]), 10000)

    # print(max(normal_pos[:,0]), max(normal_pos[:,1]))
    if i > 100:
        for i in range(NO_OF_NORMAL_PARTICLES):
            if normal_pos[i, 0] >= max_x or normal_pos[i, 1] >= max_y or normal_pos[i, 0] < 0 or normal_pos[i, 1] < 0:
                continue

            x = normal_pos[i, 0] * (HEIGHT-1) / max_x
            y = normal_pos[i, 1] * (WIDTH-1) / max_y

            frame[int(x), int(y), :] = np.array([1, 1, 1], dtype=np.uint8) * int(normal_pos[i, 2] * 255 / max_z)

        cv2.putText(frame, "Scale x:" + str(int(max_x)) + "\nScale y:" + str(int(max_y)),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)
        out.write(frame)


toc = time.time()
print("\nGenerated Simulation:",SIM_NO,"\nTime taken:", toc-tic)
png_count = SIMULATION_TIMES
files = []
# for i in range(png_count):
#     seq = str(i)
#     file_names = '3d_v2/3d_vis_' + seq + '.png'
#     files.append(file_names)
# frames = []
# for i in files:
#     new_frame = Image.open(i)
#     frames.append(new_frame)

# frames[0].save('Newton_simulations_gifs_v2/sim' + str(SIM_NO) +'.gif', format='GIF', append_images=frames[1:], save_all = True, duration = 100, loop = 0)