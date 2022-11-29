import math, numpy as np

class Camera:
	def __init__(self, idx, img_width, img_height, fx_limit, euler_limit, t_limit, is_fisheye, fy_deviation=None, skew=None,
				 num_dist=0, distortion_limit=None, noise_scale_img_diag=None):
		"""
		@skew : is either None or in radians
		@fy_deviation : is either None (that is fx=fy) or value such that fy = [fx*(1-fy_deviation/100), fx*(1+fy_deviation/100)]
		@distortion_limit : is either None or array of size (num_tangential_dist+num_radial_dist) x 2
		@euler_limit : is 3 x 2 limit of euler angles in degrees
		@t_limit : is 3 x 2 limit of translation in meters
		"""
		assert len(fx_limit) == 2 and img_width >= 0 and img_width >= 0
		if is_fisheye and distortion_limit is not None: assert len(distortion_limit) == 4 # distortion for fisheye has only 4 parameters
		self.idx = idx
		self.img_width, self.img_height = img_width, img_height
		self.fx_min = fx_limit[0]
		self.fx_max = fx_limit[1]
		self.fy_deviation = fy_deviation
		self.img_diag = math.sqrt(img_height ** 2 + img_width ** 2)
		self.is_fisheye = is_fisheye
		self.fx, self.fy = None, None
		self.px, self.py = None, None
		self.K, self.R, self.t, self.P = None, None, None, None
		self.skew = skew
		self.distortion = None
		self.distortion_lim = distortion_limit
		self.euler_limit = np.array(euler_limit, dtype=np.float32)
		self.t_limit = t_limit
		self.noise_scale_img_diag = noise_scale_img_diag
		if idx != 0:
			assert len(euler_limit) == len(t_limit) == 3
			for i in range(3):
				assert len(euler_limit[i]) == len(t_limit[i]) == 2
				self.euler_limit[i] *= (np.pi / 180)