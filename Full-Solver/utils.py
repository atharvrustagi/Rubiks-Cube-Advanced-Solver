import numpy as np
from time import perf_counter as pf
import cv2
import pygame as pg
from Functions_3D import *
from Texts import *

clrs = {'r':np.array([255, 40, 38]), 'g':np.array([36, 255, 50]), 'y':np.array([255, 200, 100]), 
        'o':np.array([255, 150, 100]), 'b':np.array([21, 113, 243]), 'w':np.array([255, 255, 255])}
clrs_keys = list(clrs.keys())

target_colors = np.array([clrs[i] for i in clrs.keys()], dtype=np.uint8)

def init_colors():
	colors = np.zeros((54, 3), dtype=np.uint8) + 100
	for i, k in enumerate(clrs_keys):
		colors[i*9 + 4] = clrs[k]
	return colors

"""
image size = 640 x 480
img.shape = (480, 640)
cube size = 240 x 240


"""

def d_angles(keys, n, dalpha, dbeta):

	if keys[pg.K_UP] and dbeta <= np.pi/2:
		dbeta += np.pi/2/n
	elif keys[pg.K_DOWN] and dbeta >= -np.pi/2:
		dbeta -= np.pi/2/n
	elif keys[pg.K_LEFT] and dalpha <= np.pi/2:
		dalpha += np.pi/2/n
	elif keys[pg.K_RIGHT] and dalpha >= -np.pi/2:
		dalpha -= np.pi/2/n
	else:
		if dalpha>1e-3:
			dalpha -= np.pi/2/n
		elif dalpha<-1e-3:
			dalpha += np.pi/2/n

		if dbeta>1e-3:
			dbeta -= np.pi/2/n
		elif dbeta<-1e-3:
			dbeta += np.pi/2/n

	return dalpha, dbeta

def find_colors2(img):
    # slicing and taking mean for image
	color_list = np.zeros((3, 3), dtype=np.uint8)
	img = cv2.flip(img, 1)
	new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	cube_size = 80
    
	for i, y in enumerate(range(img.shape[0]//2-int(cube_size*1.5), img.shape[0]//2+cube_size, cube_size)):
	    for j, x in enumerate(range(img.shape[1]//2-int(cube_size*1.5), img.shape[1]//2+cube_size, cube_size)):
	        cv2.rectangle(img, (x, y), (x+cube_size, y+cube_size), (255, 255, 255), 2)

	        slice_mean = np.mean(new_img[y+40:y+42, x+40:x+42], axis=(0, 1))
	        color_found = np.argmin(np.sum((slice_mean.reshape(1, 3) - target_colors)**2, axis=1))
	        color_list[j, i] = color_found
	        # cv2.circle(img, (x+40, y+40), 20, slice_mean[::-1], -1)
	        color = clrs[clrs_keys[color_found]][::-1].astype(np.float64)
	        cv2.circle(img, (x+40, y+40), 20, color, -1)
	        # print(slice_mean)

	return color_list.T, img

def get_pg_image(img):
	cv2.imwrite('temp.jpg', img)
	img = pg.image.load('temp.jpg')
	return img

def read_img(frame_read, img):
	if frame_read > 0:
		return 
	# cv2 webcam read
	src, img = cam.read()
	col, img = find_colors2(img)
	# transforming to pygame image
	img = get_pg_image(img)


# drawing
"""
calculations
flat_cube:
	small_cube_size = 100x100
	midpos = 1920/4, (1080-720)/2
	start = (330, 750)
"""

def draw_flat_cube(pos, win, state):
	for j, y in enumerate(range(750, 1050, 100)):
		for i, x in enumerate(range(330, 630, 100)):
			if state<=6:
				if i==1 and j==1:
					pg.draw.rect(win, state_colors[state-1], (x, y, 100, 100))
				else:
					pg.draw.rect(win, clrs[clrs_keys[pos[j, i]]], (x, y, 100, 100))
			else:
				pg.draw.rect(win, (100, 100, 100), (x, y, 100, 100))

	for y in range(750, 1051, 100):
		pg.draw.line(win, (0,0,0), (330, y), (630, y))
	for x in range(330, 631, 100):
		pg.draw.line(win, (0,0,0), (x, 750), (x, 1050))

def draw_surface(s, v, colors, win):
	# s -> 4, 2-D coordinates in cyclic order
	pg.draw.polygon(win, colors[v], s)
	for i in range(3):
		pg.draw.line(win, (32, 32, 32), s[i], s[i+1], 3)
	pg.draw.line(win, (32, 32, 32), s[0], s[3], 3)
	# index of each square
	# t = font.render(str(v), True, (0,0,0))
	# win.blit(t, np.mean(s, axis=0)-np.array([t.get_width()/2, t.get_height()/2]))

def draw_cube(cubeparams, win):
	proj_cube, z = project_surfaces(np.copy(cubeparams['cube']), cubeparams['a'], cubeparams['b'], cubeparams['pos'])
	dc = {z[x] : x for x in range(54)}
	z.sort()

	for k in reversed(z):
		v = dc[k]
		draw_surface(proj_cube[v], v, cubeparams['colors'], win)

def blit_texts(win):
	win.blit(cam_text, (1440-cam_text.get_width()/2, 120))

def create_cube(side = 50):
	s = side
	surfaces = np.zeros((6, 3, 3, 4, 3))	# 6 faces, 3x3 squares each, 4 (x,y,z) coordinates for each squares
	# left face, centre
	surfaces[0, 1, 1] = np.array([[-3*s, -s, s], [-3*s, -s, -s], [-3*s, s, -s], [-3*s, s, s]])
	# front face, centre
	surfaces[1, 1, 1] = np.array([[-s, s, -3*s], [-s, -s, -3*s], [s, -s, -3*s], [s, s, -3*s]])
	# top face, centre
	surfaces[2, 1, 1] = np.array([[-s, -3*s, -s], [-s, -3*s, s], [s, -3*s, s], [s, -3*s, -s]])

	for i in range(3):
		for j in range(3):
			# left face
			surfaces[0, i, j] = surfaces[0, 1, 1]
			surfaces[0, i, j, :, 2] -= (i-1)*2*s
			surfaces[0, i, j, :, 1] += (j-1)*2*s

			# front face
			surfaces[1, i, j] = surfaces[1, 1, 1]
			surfaces[1, i, j, :, 0] += (i-1)*2*s
			surfaces[1, i, j, :, 1] += (j-1)*2*s

			# top face
			surfaces[2, i, j] = surfaces[2, 1, 1]
			surfaces[2, i, j, :, 0] += (i-1)*2*s
			surfaces[2, i, j, :, 2] -= (j-1)*2*s

	# right face
	surfaces[3] = surfaces[0]
	surfaces[3, ..., 0] += 6*s
	surfaces[3, ..., 2] *= -1

	# back face
	surfaces[4] = surfaces[1]
	surfaces[4, ..., 2] += 6*s
	surfaces[4, ..., 0] *= -1

	# bottom face
	surfaces[5] = surfaces[2]
	surfaces[5, ..., 1] += 6*s
	surfaces[5, ..., 0] *= -1

	return surfaces



state_color_lists = [[i for i in range(18, 27)],
					 [i for i in range(9, 18)],
					 [i for i in range(27, 36)],
					 [i for i in range(36, 45)],
					 [i for i in range(9)],
					 [51,48,45,52,49,46,53,50,47]]
state_colors = np.array([clrs['y'], clrs['g'], 
						clrs['o'], clrs['b'], 
						clrs['r'], clrs['w']])

def change_cube_params(cube_colors, col, state):
	centre = np.copy(cube_colors[state_color_lists[state-1][4]])
	cube_colors[state_color_lists[state-1]] = target_colors[col[:, ::-1].T.flatten()]
	cube_colors[state_color_lists[state-1][4]] = centre

def state_change_map(state):
	if state==1:
		return (0, -1)
	elif 2<=state<=4:
		return (1, 0)
	elif state==5:
		return (0, -1)
	elif state==6:
		return (0, 0)

# start -> blue top, yellow front(cam)
# yellow - green - orange - blue - red - white
def change_angle(state, n, a, b):
	f = state_change_map(state)
	return a+f[0]*np.pi/2/n, b+f[1]*np.pi/2/n

