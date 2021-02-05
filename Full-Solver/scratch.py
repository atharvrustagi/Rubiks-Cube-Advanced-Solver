import numpy as np
import cv2
import os
from utils import *
import pygame as pg

WINSIZE = (1300, 700)
win = pg.display.set_mode()
cam = cv2.VideoCapture(0)
cam.set(3, 960)
cam.set(4, 720)
print(cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

while 1:
	for event in pg.event.get():
		if event.type==pg.QUIT:
			os.remove('temp.jpg')
			exit()

	ret, img = cam.read()
	lis, img = find_colors2(img)
	print(lis)

	cv2.imwrite('temp.jpg', img)
	img = pg.image.load('temp.jpg')

	win.fill(clrs['w'])
	win.blit(img, (0, 0))

	pg.display.update()

