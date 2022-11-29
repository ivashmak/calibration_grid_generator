import numpy as np
from camera import Camera
from utils import RandGen
from pathlib import Path


def createConfigFile(fname, params):
	Path(fname[:fname.rfind('/')]).mkdir(exist_ok=True, parents=True)
	file = open(fname, 'w')
	def writeDict(dict_write, tab):
		for key, value in dict_write.items():
			if isinstance(value, dict):
				file.write(tab+key+':\n')
				writeDict(value, tab+'  ')
			else:
				file.write(tab+key+': '+str(value)+'\n')
		file.write('\n')
	writeDict(params, '')
	file.close()

def generateCircularCameras(save_folder):
	rand_gen = RandGen(0)
	params = {'NAME': '"circular"', 'NUM_SAMPLES': 1, 'SEED': 0, 'MAX_FRAMES': 70, 'MAX_RANDOM_ITERS': 100000, 'NUM_CAMERAS': 9, 'MIN_FRAMES_PER_CAM': 1,
		'BOARD': {'WIDTH': 9, 'HEIGHT': 7, 'SQUARE_LEN':0.08, 'T_LIMIT': [[-0.2,0.2], [-0.2,0.2], [-0.1,0.1]], 'EULER_LIMIT': [[-45, 45], [-180, 180], [-45, 45]], 'T_ORIGIN': [-0.3,0,2.2]}}

	dist = 1.1
	xs = np.arange(dist, dist*(params['NUM_CAMERAS']//4)+1e-3, dist)
	xs = np.concatenate((xs, xs[::-1]))
	xs = np.concatenate((xs, -xs))
	dist_z = 0.90
	zs = np.arange(dist_z, dist_z*(params['NUM_CAMERAS']//2)+1e-3, dist_z)
	zs = np.concatenate((zs, zs[::-1]))
	yaw = np.linspace(0, -360, params['NUM_CAMERAS']+1)[1:-1]
	for i in range(9):
		fx = rand_gen.randRange(900, 1300)
		d0 = rand_gen.randRange(4e-1, 7e-1)
		euler_limit = 'null'
		t_limit = 'null'
		if i > 0:
			euler_limit = [[0,0], [yaw[i-1], yaw[i-1]], [0,0]]
			t_limit = [[xs[i-1], xs[i-1]], [0,0], [zs[i-1], zs[i-1]]]
		params['CAMERA'+str((i+1))] = {'FX': [fx, fx], 'FY_DEVIATION': 'null', 'IMG_WIDTH': int(rand_gen.randRange(1200, 1600)), 'IMG_HEIGHT': int(rand_gen.randRange(800, 1200)),
			'EULER_LIMIT': euler_limit, 'T_LIMIT': t_limit, 'NOISE_SCALE': rand_gen.randRange(2e-4, 5e-4), 'FISHEYE': False, 'DIST': [[d0,d0], [0,0], [0,0], [0,0], [0,0]]}

	createConfigFile(save_folder+'circular.yaml', params)

def getCamerasFromCfg(cfg):
	cameras = []
	for i in range(cfg['NUM_CAMERAS']):
		cameras.append(Camera(i, cfg['CAMERA' + str(i+1)]['IMG_WIDTH'], cfg['CAMERA' + str(i+1)]['IMG_HEIGHT'],
			  cfg['CAMERA' + str(i+1)]['FX'], cfg['CAMERA' + str(i+1)]['EULER_LIMIT'], cfg['CAMERA' + str(i+1)]['T_LIMIT'],
			  cfg['CAMERA' + str(i+1)]['FISHEYE'], cfg['CAMERA' + str(i+1)]['FY_DEVIATION'],
			  noise_scale_img_diag=cfg['CAMERA' + str(i+1)]['NOISE_SCALE'], distortion_limit=cfg['CAMERA' + str(i+1)]['DIST']))
	return cameras
