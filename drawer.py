import numpy as np, matplotlib.animation as manimation, matplotlib.pyplot as plt, cv2


def getDimBox(points_):
	"""
	points_: NUM_SETS x DIM x NUM_PTS
	"""
	points = np.array(points_) if type(points_) is list else points_
	assert points.ndim == 3
	return np.array([[np.median([pts[k].min() for pts in points]), np.median([pts[k].max() for pts in points])] for k in range(points.shape[1])])

def plotPoints(ax, points_list, num_dim, dim_box=None, title='', marker='o', s=7, legend=None, save_fname='', azim=0,
			   elev=0, fontsize=15, are_pts_colorful=False):
	colors = ['red', 'green', 'blue', 'black', 'magenta', 'brown']
	if dim_box is None: dim_box = getDimBox(points_list)
	if isinstance(are_pts_colorful, bool):
		colors_pts = np.random.RandomState(0).rand(len(points_list[0][0]), 3) if are_pts_colorful else None
	else:
		colors_pts = are_pts_colorful
	for ii, points in enumerate(points_list):
		color_ = colors_pts if colors_pts is not None else colors[ii]
		if num_dim == 2:
			ax.scatter(points[0], points[1], color=color_, marker=marker, s=s)
		else:
			ax.scatter(points[0], points[1], points[2], color=color_, marker=marker, s=s)

	ax.set_xlim(dim_box[0]); ax.set_ylim(dim_box[1])
	ax.set_xlabel('x', fontsize=fontsize); ax.set_ylabel('y', fontsize=fontsize)
	if num_dim == 3:
		ax.set_zlabel('z', fontsize=fontsize)
		ax.set_zlim(dim_box[2])
		ax.view_init(azim=azim, elev=elev)
		ax.set_box_aspect((dim_box[0, 1] - dim_box[0, 0], dim_box[1, 1] - dim_box[1, 0], dim_box[2, 1] - dim_box[2, 0]))
	else:
		ax.set_aspect('equal', 'box')

	if legend is not None: ax.legend(legend)
	if title != '': ax.set_title(title, fontsize=30)
	if save_fname != '': plt.savefig(save_fname, bbox_inches='tight', pad_inches=0)

def plotAllProjections(axs, cam_points_2d, cameras, sqr, pts_color=False):
	for i in range(len(cameras)):
		axs[i // sqr, i % sqr].clear()
		plotPoints(axs[i // sqr, i % sqr], [cam_points_2d[i]], 2, dim_box=[[0, cameras[i].img_width], [0, cameras[i].img_height]], title='camera '+str(i), are_pts_colorful=pts_color)
		# plotPoints(axs[i // sqr, i % sqr], [cam_points_2d[i]], 2, title='projected points, camera '+str(i), are_pts_colorful=pts_color)
		axs[i // sqr, i % sqr].invert_yaxis()

def plotCamerasAndGrid(ax, pts_grid, cam_box, cameras, colors, dim_box, pts_color=False):
	ax_lines = [None for ii in range(len(cameras))]
	ax.clear()
	ax.set_title('Cameras and grid position', fontsize=40)
	plotPoints(ax, [pts_grid], 3, s=10, are_pts_colorful=pts_color)
	all_pts = [pts_grid]
	for ii, cam in enumerate(cameras):
		cam_box_i = cam_box.copy()
		cam_box_i[:,0] *= cam.img_width / max(cam.img_height, cam.img_width)
		cam_box_i[:,1] *= cam.img_height / max(cam.img_height, cam.img_width)
		cam_box_Rt = (cam.R @ cam_box_i.T + cam.t).T
		all_pts.append(np.concatenate((cam_box_Rt, cam.t.T)).T)

		ax_lines[ii] = ax.plot([cam.t[0,0], cam_box_Rt[0,0]], [cam.t[1,0], cam_box_Rt[0,1]], [cam.t[2,0], cam_box_Rt[0,2]], '-', color=colors[ii])[0]
		ax.plot([cam.t[0,0], cam_box_Rt[1,0]], [cam.t[1,0], cam_box_Rt[1,1]], [cam.t[2,0], cam_box_Rt[1,2]], '-', color=colors[ii])
		ax.plot([cam.t[0,0], cam_box_Rt[2,0]], [cam.t[1,0], cam_box_Rt[2,1]], [cam.t[2,0], cam_box_Rt[2,2]], '-', color=colors[ii])
		ax.plot([cam.t[0,0], cam_box_Rt[3,0]], [cam.t[1,0], cam_box_Rt[3,1]], [cam.t[2,0], cam_box_Rt[3,2]], '-', color=colors[ii])

		ax.plot([cam_box_Rt[0,0], cam_box_Rt[1,0]], [cam_box_Rt[0,1], cam_box_Rt[1,1]], [cam_box_Rt[0,2], cam_box_Rt[1,2]], '-', color=colors[ii])
		ax.plot([cam_box_Rt[1,0], cam_box_Rt[2,0]], [cam_box_Rt[1,1], cam_box_Rt[2,1]], [cam_box_Rt[1,2], cam_box_Rt[2,2]], '-', color=colors[ii])
		ax.plot([cam_box_Rt[2,0], cam_box_Rt[3,0]], [cam_box_Rt[2,1], cam_box_Rt[3,1]], [cam_box_Rt[2,2], cam_box_Rt[3,2]], '-', color=colors[ii])
		ax.plot([cam_box_Rt[3,0], cam_box_Rt[0,0]], [cam_box_Rt[3,1], cam_box_Rt[0,1]], [cam_box_Rt[3,2], cam_box_Rt[0,2]], '-', color=colors[ii])
	ax.legend(ax_lines, [str(ii) for ii in range(len(cameras))], fontsize=20)

	if dim_box is None: dim_box = getDimBox([np.concatenate((all_pts),1)])
	ax.set_xlim(dim_box[0]); ax.set_ylim(dim_box[1]); ax.set_zlim(dim_box[2])
	ax.set_box_aspect((dim_box[0, 1] - dim_box[0, 0], dim_box[1, 1] - dim_box[1, 0], dim_box[2, 1] - dim_box[2, 0]))
	ax.view_init(azim=-89, elev=-15)
	return ax

def plotAllProjectionsFig(cam_points_2d, cameras, pts_color=False):
	sqr = int(np.ceil(np.sqrt(len(cameras))))
	fig, axs = plt.subplots(sqr, sqr, figsize=(15,10));
	plotAllProjections(axs, cam_points_2d, cameras, sqr, pts_color)

def getCameraBox():
	cam_box = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]],dtype=np.float32); cam_box[:,2] = 3.0
	cam_box *= 0.15
	return cam_box

def plotCamerasAndGridFig(pts_grid, cameras, pts_color=False):
	fig = plt.figure(figsize=(13.0, 15.0))
	ax = fig.add_subplot(111, projection='3d')
	return plotCamerasAndGrid(ax, pts_grid, getCameraBox(), cameras, np.random.RandomState(0).rand(len(cameras),3), None, pts_color)

def animation2D(grid, cameras, points_2d, save_proj_animation, VIDEOS_FPS, VIDEOS_DPI, MAX_FRAMES):
	writer = manimation.writers['ffmpeg'](fps=VIDEOS_FPS)
	sqr = int(np.ceil(np.sqrt(len(cameras))))
	fig, axs = plt.subplots(sqr, sqr, figsize=(15,10))
	with writer.saving(fig, save_proj_animation, dpi=VIDEOS_DPI):
		for k, cam_points_2d in enumerate(points_2d):
			if k >= MAX_FRAMES: break
			plotAllProjections(axs, cam_points_2d, cameras, sqr, pts_color=grid.colors_grid)
			writer.grab_frame()

def animation3D(grid, cameras, points_3d, save_3d_animation, VIDEOS_FPS, VIDEOS_DPI, MAX_FRAMES):
	writer = manimation.writers['ffmpeg'](fps=VIDEOS_FPS)
	fig = plt.figure(figsize=(13.0, 15.0))
	ax = fig.add_subplot(111, projection='3d')
	cam_box = getCameraBox()
	colors = np.random.RandomState(0).rand(10,3)
	all_pts = []
	cam_pts = np.concatenate([cam.R @ cam_box.T + cam.t for cam in cameras], 1)
	for k in range(min(50, len(points_3d))):
		all_pts.append(np.concatenate((cam_pts, points_3d[k]),1))
	dim_box = getDimBox(all_pts)
	with writer.saving(fig, save_3d_animation, dpi=VIDEOS_DPI):
		for i, pts_grid in enumerate(points_3d):
			if i >= MAX_FRAMES: break
			plotCamerasAndGrid(ax, pts_grid, cam_box, cameras, colors, dim_box, pts_color=grid.colors_grid)
			writer.grab_frame()
