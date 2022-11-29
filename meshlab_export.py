import cv2, numpy as np
from utils import project
from pathlib import Path

def exportSensor2XML(file, camera, id):
	file.write('\t\t\t<sensor id="'+str(id)+'" label="camera'+str(id)+'" type="frame">\n')
	# file.write('\t\t\t\t<resolution width="'+str(camera.img_width)+'" height="'+str(camera.img_height)+'"/>\n')
	file.write('\t\t\t\t<calibration type="frame" class="adjusted">\n')
	file.write('\t\t\t\t\t<resolution width="'+str(camera.img_width)+'" height="'+str(camera.img_height)+'"/>\n')
	file.write('\t\t\t\t\t<fx>'+str(camera.fx)+'</fx>\n')
	file.write('\t\t\t\t\t<fy>'+str(camera.fy)+'</fy>\n')
	file.write('\t\t\t\t\t<cx>'+str(camera.px)+'</cx>\n')
	file.write('\t\t\t\t\t<cy>'+str(camera.py)+'</cy>\n')
	file.write('\t\t\t\t\t<k1>'+str(camera.distortion[0,0])+'</k1>\n')
	file.write('\t\t\t\t\t<k2>'+str(camera.distortion[0,1])+'</k2>\n')
	# file.write('\t\t\t\t\t<k3>'+str(camera.distortion[0,4])+'</k3>\n')
	file.write('\t\t\t\t\t<p1>'+str(camera.distortion[0,2])+'</p1>\n')
	file.write('\t\t\t\t\t<p2>'+str(camera.distortion[0,3])+'</p2>\n')
	file.write('\t\t\t\t</calibration>\n')
	file.write('\t\t\t</sensor>\n')

def exportCamera2XML(file, K, R, t, label, id):
	RT = np.concatenate((np.concatenate((R, t[:,None]), 1), np.array([[0, 0, 0, 1]])))
	RT = np.linalg.inv(RT)
	# pdb.set_trace()
	file.write('\t\t\t<camera id="'+str(id)+'" label="'+label+'" sensor_id="'+str(id)+'" enabled="true">\n')
	file.write('\t\t\t\t<transform>' + np.array2string(RT.flatten(), precision=5, separator=' ')[1:-1] + '</transform>\n')
	file.write('\t\t\t</camera>\n')

def saveDataMeshlab(cameras, points_3d, save_folder):
	"""
	Steps in Meshlab to load example:
	1) File -> Import Rasters -> proj_cam_0.png, proj_cam_1.png (IMPORTANT: keep order of rasters!)
	2) File -> Import Mesh -> grid.xyz
	3) On right panel: Import cameras for active rasters from file -> Select cameras.xml
	4) Render -> Show Camera
	5) On right panel: Show Camera: Select Camera Scale Method to 'Fixed Factor' and change 'Scale Factor' to suitable one.
	6) Try 'Show Current Raster Mode' on top panel.
	"""
	Path(save_folder).mkdir(parents=True, exist_ok=True)
	np.savetxt(save_folder+'grid.xyz', points_3d.T, delimiter=' ')
	for i in range(len(cameras)):
		proj_img = np.ones((cameras[i].img_height, cameras[i].img_width, 3), dtype=np.uint8)*255
		points_proj = project(cameras[i].K, cameras[i].R, cameras[i].t.flatten(), cameras[i].distortion, points_3d, cameras[i].is_fisheye).T
		for p in points_proj: cv2.circle(proj_img, (int(p[0]), int(p[1])), 8, (255, 0, 0), -1)
		cv2.imwrite(save_folder+'proj_cam_'+str(i)+'.png', proj_img)
	xml_file = open(save_folder+'cameras.xml', 'w')
	xml_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
	xml_file.write('<document version="1.2.0">\n')
	xml_file.write('\t<chunk>\n')
	xml_file.write('\t\t<sensors>\n')
	for i in range(len(cameras)):
		exportSensor2XML(xml_file, cameras[i], i)
	xml_file.write('\t\t</sensors>\n')
	xml_file.write('\t\t<cameras>\n')
	for i in range(len(cameras)):
		exportCamera2XML(xml_file, cameras[i].K, cameras[i].R, cameras[i].t.flatten(), 'proj_cam_'+str(i)+'.png', i)
	xml_file.write('\t\t</cameras>\n')
	xml_file.write('\t</chunk>\n')
	xml_file.write('</document>\n')