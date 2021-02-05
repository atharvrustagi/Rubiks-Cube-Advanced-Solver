import numpy as np

Zv = 1000
f = 960/np.tan(np.pi/4)

def project_surfaces(cube, alpha, beta, pos):
	h = (cube[..., 0]**2 + cube[..., 2]**2)**0.5
	a = np.arctan(cube[..., 2]/(cube[..., 0] + 1e-8)) - alpha
	c = np.where(cube[..., 0]>=0, 1, -1)
	cube[..., 0] = c*h*np.cos(a)
	cube[..., 2] = c*h*np.sin(a)

	h = (cube[..., 1]**2 + cube[..., 2]**2)**0.5
	a = np.arctan(cube[..., 2]/(cube[..., 1] + 1e-8)) - beta
	c = np.where(cube[..., 1]>=0, 1, -1)
	cube[..., 1] = c*h*np.cos(a)
	cube[..., 2] = c*h*np.sin(a)

	z = np.mean(cube[..., 2], axis=3).reshape(54)
	ret = (f*cube[..., :2]/(Zv+cube[..., 2:])).reshape(54, 4, 2)
	ret[..., 0] += pos[0]
	ret[..., 1] += pos[1]
	return ret, z

