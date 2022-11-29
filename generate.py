import argparse, yaml, numpy as np, traceback, pdb, sys, math, cv2, matplotlib.pyplot as plt, copy, os
from config.config import getCamerasFromCfg, generateCircularCameras
from drawer import plotPoints, plotAllProjectionsFig, plotCamerasAndGridFig, plotCamerasAndGrid, animation2D, animation3D
from meshlab_export import saveDataMeshlab
from utils import RandGen, project, insideImage, eul2rot, areAllInsideImage, insideImageMask, projectCamera, export2JSON
from pathlib import Path
from grid import CheckerBoard

def generateCameras(cameras, rand_gen, EPS=1e-10):
	"""
	generates camera parameters, intrinsics, extrinsics
	"""
	for i in range(len(cameras)):
		cameras[i].t = np.zeros((3, 1))
		if cameras[i].idx == 0:
			cameras[i].R = np.identity(3)
		else:
			angles = [0, 0, 0]
			for k in range(3):
				if abs(cameras[i].t_limit[k][0] - cameras[i].t_limit[k][1]) < EPS:
					cameras[i].t[k] = cameras[i].t_limit[k][0]
				else:
					cameras[i].t[k] = rand_gen.randRange(cameras[i].t_limit[k][0], cameras[i].t_limit[k][1])

				if abs(cameras[i].euler_limit[k][0] - cameras[i].euler_limit[k][1]) < EPS:
					angles[k] = cameras[i].euler_limit[k][0]
				else:
					angles[k] = rand_gen.randRange(cameras[i].euler_limit[k][0], cameras[i].euler_limit[k][1])

			cameras[i].R = eul2rot(angles)

		if abs(cameras[i].fx_min - cameras[i].fx_max) < EPS:
			cameras[i].fx = cameras[i].fx_min
		else:
			cameras[i].fx = rand_gen.randRange(cameras[i].fx_min, cameras[i].fx_max)
		if cameras[i].fy_deviation is None:
			cameras[i].fy = cameras[i].fx
		else:
			cameras[i].fy = rand_gen.randRange((1 - cameras[i].fy_deviation) * cameras[i].fx,
									  (1 + cameras[i].fy_deviation) * cameras[i].fx)

		cameras[i].px = int(cameras[i].img_width / 2.0) + 1
		cameras[i].py = int(cameras[i].img_height / 2.0) + 1
		cameras[i].K = np.array([[cameras[i].fx, 0, cameras[i].px], [0, cameras[i].fy, cameras[i].py], [0, 0, 1]], dtype=float)
		if cameras[i].skew is not None: cameras[i].K[0, 1] = np.tan(cameras[i].skew) * cameras[i].K[0, 0]
		cameras[i].P = cameras[i].K @ np.concatenate((cameras[i].R, cameras[i].t), 1)

		if cameras[i].distortion_lim is not None:
			cameras[i].distortion = np.zeros((1, len(cameras[i].distortion_lim)))
			for k, lim in enumerate(cameras[i].distortion_lim):
				cameras[i].distortion[0,k] = rand_gen.randRange(lim[0], lim[1])
		else:
			cameras[i].distortion = np.zeros((1, 5)) # opencv is using 5 values distortion by default

def findGridOrigin(grid, cameras):
	origin = None
	# box is simplified calibration grid to speed-up optimization
	box = np.array([[0, grid.square_len * (grid.w - 1), 0, grid.square_len * (grid.w - 1)],
					[0, 0, grid.square_len * (grid.h - 1), grid.square_len * (grid.h - 1)],
					[0, 0, 0, 0]])
	try:
		import torch, pytorch3d, pytorch3d.transforms
		has_pytorch = True
	except:
		has_pytorch = False

	if has_pytorch:
		"""
		find origin such that is minimizes distance between projected points and corners of the image
		"""
		rot_angles = torch.zeros(3, requires_grad=True)
		origin = torch.ones((3,1), requires_grad=True)
		optimizer = torch.optim.Adam([rot_angles, origin], lr=5e-3)
		Ps = torch.tensor(np.stack([cam.K @ np.concatenate((cam.R, cam.t), 1) for cam in cameras]), dtype=torch.float32)
		rot_conv = 'XYZ'
		grid_pattern = torch.tensor(box, dtype=Ps.dtype)
		corners = torch.tensor([[[0, 0], [0, cam.img_height], [cam.img_width, 0], [cam.img_width, cam.img_height]] for cam in cameras], dtype=Ps.dtype).transpose(-1,-2)
		loss_fnc = torch.nn.HuberLoss()
		lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-4, factor=0.8, patience=10)
		prev_loss = 1e10
		torch.autograd.set_detect_anomaly(True)
		for it in range(500):
			pts_grid = pytorch3d.transforms.euler_angles_to_matrix(rot_angles, rot_conv) @ grid_pattern + origin
			pts_proj = Ps[:,:3,:3] @ pts_grid[None,:] + Ps[:,:,[-1]]
			pts_proj = pts_proj[:, :2] / (pts_proj[:, [2]]+1e-15)

			loss = num_wrong = 0
			for i, proj in enumerate(pts_proj):
				if not areAllInsideImage(pts_proj[i], cameras[i].img_width, cameras[i].img_height):
					loss += loss_fnc(corners[i], pts_proj[i])
					num_wrong += 1
			if num_wrong > 0:
				loss /= num_wrong
				loss.backward(); optimizer.step(); lr_scheduler.step(loss)
				if origin[2] < 0:
					with torch.no_grad(): origin[2] = 2.0
				# if origin[2] > MAX_DEPTH:
				# 	with torch.no_grad(): origin[2] = MAX_DEPTH.0

				if it % 5 == 0:
					print('iter', it, 'loss %.2E' % loss)
					if abs(prev_loss - loss) < 1e-10:
						break
					prev_loss = loss.item()
			else:
				print('all points inside')
				break
		origin = origin.detach().numpy().reshape(3,1)
	else:
		"""
		find origin such that projected points are visible in the image
		"""
		max_points_visible = 0
		for z in np.arange(0.25, 10, .5):
			min_x1, max_x1 = -z * cameras[0].px / cameras[0].fx, (cameras[0].img_width * z - z * cameras[0].px) / cameras[0].fx
			min_y1, max_y1 = -z * cameras[0].py / cameras[0].fy, (cameras[0].img_height * z - z * cameras[0].py) / cameras[0].fy
			min_x2, max_x2 = -z * cameras[0].px / cameras[0].fx - box[0, 1], (cameras[0].img_width * z - z * cameras[0].px) / cameras[0].fx - box[0, 1]
			min_y2, max_y2 = -z * cameras[0].py / cameras[0].fy - box[1, 2], (cameras[0].img_height * z - z * cameras[0].py) / cameras[0].fy - box[1, 2]
			min_x = max(min_x1, min_x2); min_y = max(min_y1, min_y2)
			max_x = min(max_x1, max_x2); max_y = min(max_y1, max_y2)
			# print(z, min_x, max_x, min_y, max_y)
			if max_x < min_x or max_y < min_y: continue
			for x in np.linspace(min_x, max_x, 40):
				for y in np.linspace(min_y, max_y, 40):
					pts = box + np.array([[x], [y], [z]])
					num_pts_visible = 0
					# fig2d, axs = plt.subplots(1, 2);
					for i in range(len(cameras)):
						pts_proj = projectCamera(cameras[i], pts)
						# plotPoints(axs[i], [pts_proj], 2)
						# plotPoints(axs[i], [cam_points_2d[i]], 2, dim_box=[[0, cameras[i].img_width], [0, cameras[i].img_height]])

						visible_pts = insideImage(pts_proj, cameras[i].img_width, cameras[i].img_height)
						# print(i,')',x, y, z, 'not visible, total', visible_pts, '/', pts_proj.shape[1])
						num_pts_visible += visible_pts
					# plt.show()
					if num_pts_visible > max_points_visible:
						max_points_visible = num_pts_visible
						print('num pts visible', max_points_visible, x, y, z)
						origin = np.array([[x], [y], [z]])
		if origin is None:
			print('Warning, failed to find suitable origin! set to zero')
			origin = np.zeros((3,1))
	print('found origin', origin)
	return origin

def generateRTgrid(rand_gen, grid):
	"""
	generates rotation and translation of the calibration grid
	"""
	R_grid = eul2rot([ rand_gen.randRange(grid.euler_limit[0][0], grid.euler_limit[0][1]),
					rand_gen.randRange(grid.euler_limit[1][0], grid.euler_limit[1][1]),
					rand_gen.randRange(grid.euler_limit[2][0], grid.euler_limit[2][1])])
	t_grid = np.array([[rand_gen.randRange(grid.t_limit[0][0], grid.t_limit[0][1])],
					[rand_gen.randRange(grid.t_limit[1][0], grid.t_limit[1][1])],
					[rand_gen.randRange(grid.t_limit[2][0], grid.t_limit[2][1])]])
	return R_grid, t_grid

def generateAll(cameras, grid, num_frames, rand_gen, MAX_RAND_ITERS=10000, MIN_FRAMES_PER_CAM=1, save_proj_animation=None, save_3d_animation=None,
				save_dir_render=None, VIDEOS_FPS=5, VIDEOS_DPI=250, MAX_FRAMES=100):
	"""
	output: points 2d and 3d, NUM_FRAMES x NUM_CAMERAS x NUM_POINTS x 2/3
	"""
	if save_dir_render is not None:
		from shapely.geometry import Point
		from shapely.geometry.polygon import Polygon

	generateCameras(cameras, rand_gen)
	# if origin of calibration grid is provided then use it, otherwise find best one that fits all projections
	if grid.t_origin is not None:
		origin = grid.t_origin
	else:
		origin = findGridOrigin(grid, cameras)

	points_grid = grid.pattern + origin
	points_grid_mean = points_grid.mean(-1)[:,None]
	if save_dir_render:
		points_grid_cells = grid.cells_3d + origin
		points_grid_mean_cells = points_grid_cells.mean(-1)[:,None]

	points_2d, points_3d = [], []
	valid_frames_per_camera = np.zeros(len(cameras))
	# plotCamerasAndGridFig(points_grid, cameras, pts_color=grid.colors_grid); plt.show()
	for frame in range(MAX_RAND_ITERS):
		R_grid, t_grid = generateRTgrid(rand_gen, grid)
		pts_grid = R_grid @ (points_grid - points_grid_mean) + points_grid_mean + t_grid
		cam_points_2d = [projectCamera(cam, pts_grid) for cam in cameras]
		
		"""
		######### plot normals ########
		grid_normal = 10*np.cross(pts_grid[:,grid.w] - pts_grid[:,0], pts_grid[:,grid.w-1] - pts_grid[:,0])
		ax = plotCamerasAndGridFig(pts_grid, cameras, pts_color=grid.colors_grid);
		pts = np.stack((pts_grid[:,0], pts_grid[:,0]+grid_normal))
		ax.plot(pts[:,0], pts[:,1], pts[:,2], 'r-')
		for ii, cam in enumerate(cameras):
			pts = np.stack((cam.t.flatten(), cam.t.flatten()+cam.R[2]))
			ax.plot(pts[:,0], pts[:,1], pts[:,2], 'g-')
			print(ii, np.arccos(grid_normal.dot(cam.R[2]) / np.linalg.norm(grid_normal))*180/np.pi, np.arccos((-grid_normal).dot(cam.R[2]) / np.linalg.norm(grid_normal))*180/np.pi)
		plotAllProjectionsFig(np.stack(cam_points_2d), cameras, pts_color=grid.colors_grid);
		plt.show()
		###############################
		"""

		valid_proj = np.zeros(len(cameras), dtype=bool)
		for cam_idx in range(len(cameras)):
			if not grid.isProjectionValid(cam_points_2d[cam_idx]):
				cam_points_2d[cam_idx] = -np.ones_like(cam_points_2d[cam_idx])
			elif cameras[cam_idx].noise_scale_img_diag is not None:
				cam_points_2d[cam_idx] += rand_gen.rand_gen.normal(0, cameras[cam_idx].img_diag * cameras[cam_idx].noise_scale_img_diag, cam_points_2d[cam_idx].shape)
				valid_proj[cam_idx] = True

		pts_inside_camera = np.zeros(len(cameras), dtype=bool)
		for ii, pts_2d in enumerate(cam_points_2d):
			mask = insideImageMask(pts_2d, cameras[ii].img_width, cameras[ii].img_height)
			pts_inside_camera[ii] = mask.all()

		if pts_inside_camera.sum() >= 2:
			if save_dir_render is not None:
				pts_grid_cells = R_grid @ (points_grid_cells - points_grid_mean_cells) + points_grid_mean_cells + t_grid
				points_2d_cells = [projectCamera(cam, pts_grid_cells) for cam in cameras]
				for idx, pts in enumerate(points_2d_cells):
					img = 125*np.ones((cameras[idx].img_height, cameras[idx].img_width, 3), dtype=np.uint8)
					def save_img():
						cv2.imwrite(save_dir_render+'cam_'+str(idx)+'/'+('%014d' % len(points_2d))+'.jpg', img)

					if not pts_inside_camera[idx]:
						save_img()
						continue

					pts = np.array(pts, dtype=np.float32)
					for ii, (pol_idxs, pol_color) in enumerate(zip(grid.polygons_idxs, grid.polygons_colors)):
						poly = Polygon(pts[:,pol_idxs].T)
						chull = np.array(np.round(cv2.convexHull(pts[:,pol_idxs].T)[:,0]), dtype=int)
						min_x, min_y = np.array(chull.min(0), dtype=int)
						max_x, max_y = np.array(chull.max(0), dtype=int)
						min_x = max(0, min_x)
						min_y = max(0, min_y)
						max_x = min(cameras[idx].img_width, max_x)
						max_y = min(cameras[idx].img_height, max_y)
						for i in range(min_x, max_x, 1):
							for j in range(min_y, max_y, 1):
								if poly.contains(Point(i, j)):
									img[j, i] = np.ones(3, dtype=np.uint8)*255*pol_color

					# plt.figure();
					# plt.imshow(img)
					# plt.title('image '+str(idx));
					# plt.scatter(cam_points_2d[idx][0], cam_points_2d[idx][1])
					# plt.show()
					save_img()

			valid_frames_per_camera += np.array(pts_inside_camera, int)
			print('valid frames per camera', valid_frames_per_camera)
			points_2d.append(np.stack(cam_points_2d))
			points_3d.append(pts_grid)

			# plotAllProjectionsFig(np.stack(cam_points_2d), cameras, pts_color=grid.colors_grid)
			# plotCamerasAndGridFig(pts_grid, cameras, pts_color=grid.colors_grid)
			# plt.show()
			if len(points_2d) >= num_frames and (valid_frames_per_camera >= MIN_FRAMES_PER_CAM).all():
				print('tried iterations', frame)
				break

	if (valid_frames_per_camera < MIN_FRAMES_PER_CAM).any():
		print('Failed to generate points that satify min frames constraint!')
		return None, None

	if save_proj_animation is not None: animation2D(grid, cameras, points_2d, save_proj_animation, VIDEOS_FPS, VIDEOS_DPI, MAX_FRAMES)
	if save_3d_animation is not None: animation3D(grid, cameras, points_3d, save_3d_animation, VIDEOS_FPS, VIDEOS_DPI, MAX_FRAMES)
	print('number of found frames', len(points_2d))
	return np.stack(points_2d), np.stack(points_3d)


def main(cfg_name, save_folder, save_animation, save_rendering, export_to_meshlab):
	assert os.path.exists(cfg_name), 'config file ('+ cfg_name +') does not exist exist!'
	cfg = yaml.safe_load(open(cfg_name, 'r'))
	print(cfg)
	for trial in range(cfg['NUM_SAMPLES']):
		save_folder = save_folder + cfg['NAME'] + "_sample_"+str(trial)+'/'
		Path(save_folder).mkdir(exist_ok=True, parents=True)

		checkerboard = CheckerBoard(cfg['BOARD']['WIDTH'], cfg['BOARD']['HEIGHT'], cfg['BOARD']['SQUARE_LEN'], cfg['BOARD']['EULER_LIMIT'], cfg['BOARD']['T_LIMIT'], cfg['BOARD']['T_ORIGIN'])
		cameras = getCamerasFromCfg(cfg)
		if save_animation:
			Path(save_folder+'animation/').mkdir(exist_ok=True, parents=True)
			projection_file = save_folder + 'animation/plots_projections.mp4'
			grid_file = save_folder + 'animation/grid_cameras.mp4'
		else:
			projection_file, grid_file = None, None
		if save_rendering:
			render_save_dir = save_folder+'render/'
			for cam in range(len(cameras)):
				Path(render_save_dir+'cam_'+str(cam)).mkdir(exist_ok=True, parents=True)
		else:
			render_save_dir = None

		points_2d, points_3d = generateAll(cameras, checkerboard, cfg['MAX_FRAMES'], RandGen(cfg['SEED']), cfg['MAX_RANDOM_ITERS'], cfg['MIN_FRAMES_PER_CAM'], projection_file, grid_file, render_save_dir)
		if points_2d is None:
			continue
		for i in range(len(cameras)):
			print('Camera', i)
			print('K', cameras[i].K)
			print('R', cameras[i].R)
			print('t', cameras[i].t.flatten())
			print('distortion', cameras[i].distortion.flatten())
			print('-----------------------------')

		export2JSON(checkerboard.pattern, points_2d, cameras, save_folder+cfg['NAME']+'.json')

		if export_to_meshlab:
			save_folder_mesh = save_folder+'meshlab/'
			Path(save_folder_mesh).mkdir(exist_ok=True, parents=True)
			import pymeshlab
			saveDataMeshlab(cameras, points_3d[0], save_folder_mesh)
			ms = pymeshlab.MeshSet()
			ms.load_new_mesh(save_folder_mesh + 'grid.xyz')
			for i in range(len(cameras)):
				ms.load_new_raster(save_folder_mesh + 'proj_cam_' + str(i) + '.png')
			ms.load_active_raster_cameras(**{'importfile': save_folder_mesh + 'cameras.xml'})
			if save_folder_mesh[0] != '/':
				print('Warning! To save meshlab file the save directory must start with a full path, i.e., "/"')
			ms.save_project(save_folder_mesh + "meshlab_demo.mlp")

if __name__ == '__main__':
	try:
		parser = argparse.ArgumentParser()
		parser.add_argument('--cfg', type=str, required=True, help='path to config file, e.g., config_cv_test.yaml')
		parser.add_argument('--output_folder', type=str, default='', help='output folder')
		parser.add_argument('--save_animation', type=int, default=1, help='1 -- save animation, otherwise 0')
		parser.add_argument('--save_rendering', type=int, default=1, help='1 -- save rendering, otherwise 0')
		parser.add_argument('--export_to_meshlab', type=int, default=1, help='1 -- save to meshlab file, otherwise 0')

		params, _ = parser.parse_known_args()
		main(params.cfg, params.output_folder, params.save_animation != 0, params.save_rendering != 0, params.export_to_meshlab != 0)
	except:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
