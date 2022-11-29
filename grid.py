import numpy as np

class Grid:
	def __init__(self, w, h, square_len, euler_limit, t_limit, t_origin=None):
		assert w >= 0 and h >= 0 and square_len >= 0
		assert len(euler_limit) == len(t_limit) == 3
		self.w = w
		self.h = h
		self.square_len = square_len
		self.t_limit = t_limit
		self.euler_limit = np.array(euler_limit, dtype=np.float32)
		colors = [[1,0,0], [0,1,0], [0,0,0], [0,0,1]]
		self.colors_grid = np.zeros((w*h, 3))
		self.t_origin = np.array(t_origin, dtype=np.float32)[:,None] if t_origin is not None else None
		for i in range(h):
			for j in range(w):
				if j <= w // 2 and i <= h // 2: color = colors[0]
				elif j <= w // 2 and i > h // 2: color = colors[1]
				elif j > w // 2 and i <= h // 2: color = colors[2]
				else: color = colors[3]
				self.colors_grid[i*w+j] = color
		for i in range(3):
			assert len(euler_limit[i]) == len(t_limit[i]) == 2
			self.euler_limit[i] *= (np.pi / 180)

	def isProjectionValid(self, pts_proj):
		"""
		projection is valid, if x coordinate of left top corner point is smaller than x of bottom right point, ie do not allow 90 deg rotation of 2D board
		also, if x coordinate of left bottom corner is smaller than x coordinate on top right corner, ie do not allow flip
		pts_proj : 2 x N
		"""
		assert pts_proj.ndim == 2 and pts_proj.shape[0] == 2
		return pts_proj[0,0] < pts_proj[0,-1] and pts_proj[0,(self.h-1)*self.w] < pts_proj[0,self.w-1]

class CircleBoard(Grid):
	def __init__(self, w, h, square_len, euler_limit, t_limit, t_origin=None):
		super().__init__(w, h, square_len, euler_limit, t_limit, t_origin)
		self.pattern = []
		for row in range(h):
			for col in range(w):
				if row % 2 == 1:
					self.pattern.append([(col+.5)*square_len, square_len*(row//2+.5), 0])
				else:
					self.pattern.append([col*square_len, (row//2)*square_len, 0])
		self.pattern = np.array(self.pattern, dtype=np.float32).T

class CheckerBoard(Grid):
	def __init__(self, w, h, square_len, euler_limit, t_limit, t_origin=None):
		super().__init__(w, h, square_len, euler_limit, t_limit, t_origin)
		self.pattern = np.zeros((w * h, 3), np.float32)
		self.pattern[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * square_len  # only for (x,y,z=0)
		self.pattern = self.pattern.T

		self.cells = []
		self.polygons_colors = []
		self.polygons_idxs = []
		# pattern: 3 x N
		# first row: 0, 1, 2, ..., w-1, 0, 1, 2, ..., w-1, ...
		# second row 0, 0, ..., 0, 1, 1, ..., 1, ..., h-1, h-1, ...,h-1, each value repeats w times
		# last row is zeros 
		num_splits = 5
		self.colors_cells = []
		num_pts_total = 0
		for i in range(1, h):
			for j in range(1, w):
				row_top = np.stack((np.linspace(j-1, j, num_splits), (i-1)*np.ones(num_splits)))
				row_bot = np.stack((np.linspace(j-1, j, num_splits),     i*np.ones(num_splits)))
				col_lef = np.stack(((j-1)*np.ones(num_splits), np.linspace(i-1, i, num_splits)))
				col_rig = np.stack((    j*np.ones(num_splits), np.linspace(i-1, i, num_splits)))
				cell = np.concatenate((row_top, row_bot, col_lef, col_rig), 1)*square_len
				self.cells.append(cell)
				self.polygons_idxs.append(np.arange(num_pts_total, num_pts_total+cell.shape[-1], 1))
				if (i % 2 and j % 2) or (i % 2 == 0 and j % 2 == 0):
					self.polygons_colors.append(1)
					self.colors_cells.append(np.ones(cell.shape[-1]))
				else:
					self.polygons_colors.append(0)
					self.colors_cells.append(np.zeros(cell.shape[-1]))
				num_pts_total += cell.shape[-1]
		self.cells = np.concatenate(self.cells, -1)
		self.cells_3d = np.concatenate((self.cells, np.zeros((1, self.cells.shape[-1]))), 0)
		self.colors_cells = np.concatenate(self.colors_cells)
