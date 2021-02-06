import pygame as pg

pg.font.init()
font = pg.font.SysFont("georgia", 20)
font2 = pg.font.SysFont("georgia", 30)
BLACK = (0, 0, 0)

cam_text = font.render("CAMERA", True, BLACK)

scanning_cube_text = font2.render("Scanning the cube...", True, BLACK)
click_anywhere_text = font2.render("Click anywhere to move to the next face", True, BLACK)

state_text = [
	font2.render("Show Yellow face with Blue face on top", True, BLACK), 
	font2.render("Show Green face with Yellow face on top", True, BLACK), 
	font2.render("Show Orange face with Yellow face on top", True, BLACK), 
	font2.render("Show Blue face with Yellow face on top", True, BLACK), 
	font2.render("Show Red face with Yellow face on top", True, BLACK), 
	font2.render("Show White face with Red face on top", True, BLACK)	]

solving_text = font2.render("Solving, please wait (Max 10 seconds)", True, BLACK)

invalid_text = font2.render("The cube is in an invalid state", True, BLACK)
scan_again_text = font2.render("Scan the cube again", True, BLACK)

solution_found_text = font2.render("Solution Found!", True, BLACK)
click_anywhere2_text = font2.render("Click anywhere to execute", True, BLACK)
