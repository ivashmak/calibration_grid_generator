import os, numpy as np, math, cv2, sys, pdb, traceback, json
from pathlib import Path

class RandGen:
	def __init__(self, seed = 0):
		self.rand_gen = np.random.RandomState(seed)
	def randRange(self, min_v, max_v):
		return self.rand_gen.rand(1).item() * (max_v - min_v) + min_v

def project(K, R, t, dist, pts_3d, is_fisheye):
	if is_fisheye:
		pts_2d = cv2.fisheye.projectPoints(pts_3d.T[None,:], cv2.Rodrigues(R)[0], t, K, dist.flatten())[0].reshape(-1,2).T
	else:
		pts_2d = cv2.projectPoints(pts_3d, R, t, K, dist)[0].reshape(-1,2).T
	return pts_2d

def projectCamera(camera, pts_3d):
	return project(camera.K, camera.R, camera.t, camera.distortion, pts_3d, camera.is_fisheye)

def eul2rot(theta): # [x y z]
	# https://learnopencv.com/rotation-matrix-to-euler-angles/
	R_x = np.array([[1,		 0,				  0				   ],
					[0,		 math.cos(theta[0]), -math.sin(theta[0]) ],
					[0,		 math.sin(theta[0]), math.cos(theta[0])  ]])
	R_y = np.array([[math.cos(theta[1]),	0,	  math.sin(theta[1])  ],
					[0,					 1,	  0				   ],
					[-math.sin(theta[1]),   0,	  math.cos(theta[1])  ]])
	R_z = np.array([[math.cos(theta[2]),	-math.sin(theta[2]),	0],
					[math.sin(theta[2]),	math.cos(theta[2]),	 0],
					[0,					 0,					  1]])
	return np.dot(R_z, np.dot(R_y, R_x))

def insideImageMask(pts, w, h):
	return np.logical_and(np.logical_and(pts[0] < w, pts[1] < h), np.logical_and(pts[0] > 0, pts[1] > 0))

def insideImage(pts, w, h):
	return insideImageMask(pts, w, h).sum()

def areAllInsideImage(pts, w, h):
	return insideImageMask(pts, w, h).all()

def export2JSON(pattern_points, image_points, cameras, json_file):
	image_points = image_points.transpose(1,0,3,2)
	image_points_list = [[] for _ in cameras]
	is_fisheye = [c.is_fisheye for c in cameras]
	image_sizes = [[c.img_width, c.img_height] for c in cameras]
	distortions = [c.distortion.tolist() for c in cameras]
	R = [c.R.tolist() for c in cameras]
	T = [c.t.tolist() for c in cameras]
	K = [c.K.tolist() for c in cameras]
	for c in range(len(image_points)):
		for f in range(len(image_points[c])):
			if areAllInsideImage(image_points[c][f], cameras[c].img_width, cameras[c].img_height):
				image_points_list[c].append(image_points[c][f].tolist())
			else:
				image_points_list[c].append([])
	json.dump({'K': K, 'R': R, 'T': T, 'distortion': distortions, 'object_points': pattern_points.tolist(), 'image_points': image_points_list, 'image_sizes': image_sizes, 'is_fisheye': is_fisheye}, open(json_file, 'w'))
