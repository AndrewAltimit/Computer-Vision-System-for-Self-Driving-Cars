import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

# PARAMETERS
MOVING_THRESHOLD = 0.5
MAX_CLUSTER_DISTANCE = 100


# Find the distance between a point and a line (a, b, c)
def get_distance_to_line(a, b, c, x, y):
	return abs((a * x) + (b * y) + c) / np.sqrt((a ** 2) + (b ** 2))
	
	
# Return the line (a, b, c) between 2 points
def get_line(x1, y1, x2, y2):
	a = (y1 - y2)
	b = (x2 - x1)
	c = x1 * y2 - x2 * y1
	return a, b, c
	
	
# Draw the Focus of Expansion: https://docs.opencv.org/3.2.0/da/de9/tutorial_py_epipolar_geometry.html
def draw_foe(img1,lines,pts1,pts2):
	r,col = img1.shape
	img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
	
	# draw the motion vector
	mask = np.zeros_like(img1)
	color = (0, 50, 250)
	for i,(pt1,pt2) in enumerate(zip(pts1,pts2)):
		a,b = pt1.ravel()
		c,d = pt2.ravel()
		img1 = cv2.line(img1, (a,b),(c,d), color, 2)

	return img1
	
	
# Determine independently moving points with a moving camera
def draw_independently_moving_points_movingcam(img1, lines, pts1, pts2, intersection1, intersection2):
	r,col = img1.shape
	img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
	cx1, cy1 = intersection1
	cx2, cy2 = intersection2
	
	moving_color = (255, 0, 0)
	stable_color = (0, 255, 0)
	moving_point_indices = []
	
	# Determine which keypoint pairs have a significant difference in movement from the FoE movement
	for i, (r, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
		a, b, c = r
		x0,y0 = map(int, [0, -c/b ])
		x1,y1 = map(int, [col, -(c+a*col)/b ])
		
		# 1. Get a b c from pt1 and pt2
		a, b, c = get_line(pt1[0], pt1[1], pt2[0], pt2[1])
		
		# 2. Get a b c from center1 and center2 of the epipolar line intersections
		a_c, b_c, c_c = get_line(cx1, cy1, cx2, cy2)
		
		# 3. The difference between a, b, c from 1 and a, b, c from 2 will indicate if it is a unique motion (not from FoE)
		offset = abs(a - a_c) + abs(b - b_c)
		
		if offset  > 10:
			img1 = cv2.circle(img1,tuple(pt1),5,moving_color,-1)
			moving_point_indices.append(i)
		else:
			img1 = cv2.circle(img1,tuple(pt1),5,stable_color,-1)
	
	return img1, moving_point_indices
	
# Determine independently moving points with the camera in a fixed position
def draw_independently_moving_points_stablecam(img1, pts1, pts2):
	img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
	moving_color = (255, 0, 0)
	stable_color = (0, 255, 0)
	moving_point_indices = []
	
	# Determine which keypoint pairs have a significant difference in position
	for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
		x1, y1 = pt1
		x2, y2 = pt2
		distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
		
		# If the points have moved any noticeable amount, then the object is independently moving due to the stable camera
		if distance  > 1:
			img1 = cv2.circle(img1,tuple(pt1),5,moving_color,-1)
			moving_point_indices.append(i)
		else:
			img1 = cv2.circle(img1,tuple(pt1),5,stable_color,-1)
	
	return img1, moving_point_indices
	

# Cluser a group of moving points for a given image
def cluster_moving_objects(img1, moving_point_indices, pts1):
	# Get Linked Points
	# node list [a b c d e f g]
	# connected  ^[b f g]
	linked_points = []
	for i in range(len(moving_point_indices)):
		ind_a = moving_point_indices[i]
		x_A1, y_A1 = pts1[ind_a]
		linked_points.append(set())
		linked_points[-1].add(i)
		
		# Compare with all other points
		for j in range(len(moving_point_indices)):
			ind_b = moving_point_indices[j]
			
			x_B2, y_B2 = pts1[ind_b]
			distance = np.sqrt((x_A1 - x_B2) ** 2 + (y_A1 - y_B2) ** 2)
			
			if distance < MAX_CLUSTER_DISTANCE:
				linked_points[-1].add(j)
				
				
	# Find clusters by searching all links to any given point
	groups = set()
	for i in range(len(linked_points)):
		group = set(linked_points[i])
		next_interation_pairs = group.copy()
		# Continue searching matching links until no new links are found
		while len(next_interation_pairs) > 0:
			current_pairs = next_interation_pairs
			next_interation_pairs = set()
			for j in current_pairs:
				other_pairs = linked_points[j]
				new_entries = set(other_pairs) - set(group)
				next_interation_pairs = next_interation_pairs.union(new_entries)
				group = group.union(new_entries)		
			
		group = sorted(list(group))
		groups.add(tuple(group))
	
	groups = list(groups)
	
	# Display the clusters
	img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
	for i in range(len(groups)):
		color = tuple(np.random.randint(0,255,3).tolist())
		group = groups[i]
		
		# Group centerx, centery, and min/max x and y points used to draw bounding circle
		gcenterx = 0
		gcentery = 0
		min_x = pts1[moving_point_indices[group[0]]][0]
		min_y = pts1[moving_point_indices[group[0]]][1]
		max_x = pts1[moving_point_indices[group[0]]][0]
		max_y = pts1[moving_point_indices[group[0]]][1]
		
		# Draw cluster points
		for index in group:
			x, y = pts1[moving_point_indices[index]]
			img1 = cv2.circle(img1,(x, y),5,color,-1)
			gcenterx += x
			gcentery += y
			min_x = min(min_x, x)
			min_y = min(min_y, y)
			max_x = max(max_x, x)
			max_y = max(max_y, y)
			
		# Draw the bounding circle
		gcenterx /= len(group)
		gcentery /= len(group)
		if len(group) == 1:
			radius = 25
		else:
			radius = max(max_x - min_x, max_y - min_y)
		img1 = cv2.circle(img1,(int(gcenterx), int(gcentery)), int(radius),color,3)
		
	return img1

				
	
# Given a list of lines, determine the intersection point using least-squares
def find_intersection_point(lines):
	# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
	# http://infohost.nmt.edu/tcc/help/lang/python/examples/homcoord/Line-intersect.html (used least-squares but similar idea)
	a = lines[:, :2]
	b = -1 * lines[:, 2:]
	x, y = np.linalg.lstsq(a, b)[0]
	return x, y


if __name__ == "__main__":
	
	# Check if all proper input arguments exist
	if len(sys.argv) != 3:
		print("Improper number of input arguments")
		print("USAGE: main.py <in_img_1> <in_img_2>")
		sys.exit()	

	
	# Get image path from user input and read in image
	img1_gray = cv2.imread(sys.argv[1], 0)
	img2_gray = cv2.imread(sys.argv[2], 0)
	
	
	# Lucas-Kanade Optical Flow in OpenCV
	# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
	# params for ShiTomasi corner detection
	feature_params = dict( maxCorners = 100,
						   qualityLevel = 0.3,
						   minDistance = 7,
						   blockSize = 7 )
	
	# Parameters for lucas kanade optical flow
	lk_params = dict( winSize  = (15,15),
					  maxLevel = 2,
					  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	
	# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
	# Take first frame and find corners in it
	p1 = cv2.goodFeaturesToTrack(img1_gray, mask = None, **feature_params)

	# calculate optical flow
	p2, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p1, None, **lk_params)
	
	# Select good points
	pts1 = p1[st==1]
	pts2 = p2[st==1]
	
	# Determine if the camera is moving or not
	moving_count = []
	for new,old in zip(pts2,pts1):
		a,b = new.ravel()
		c,d = old.ravel()
		dist = np.sqrt((a - c)**2 + (b - d)**2)
		moving_count.append(1 * (dist > 1))
		
	percent_moving = sum(moving_count) / len(moving_count)
	print("Percent Moving: {:.2f} %".format(percent_moving * 100))
	is_moving = percent_moving > MOVING_THRESHOLD
	if is_moving:
		print("Status: Camera is moving")
	else:
		print("Status: Camera is stationary")

	# Get fundamental matrix and filter out bad points
	F, fmask = cv2.findFundamentalMat(pts1.astype(np.int32),pts2.astype(np.int32),cv2.FM_RANSAC) 
	print("Fundamental Matrix\n", F)
	pts1 = pts1[fmask.ravel()==1]
	pts2 = pts2[fmask.ravel()==1]
	
	# Find epilines corresponding in the images
	lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F).reshape(-1,3)
	lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F).reshape(-1,3)

	# drawing its lines on the left and right images
	img1_foe = draw_foe(img1_gray, lines1, pts1, pts2)
	
	if is_moving:
		# Least Squares Error - Line Intersection
		x1, y1 = find_intersection_point(lines1)
		x2, y2 = find_intersection_point(lines2)
		print("Intersection Point in first image: ({:.0f},{:.0f})".format(x1[0],y1[0]))
		print("Intersection Point in second image: ({:.0f},{:.0f})".format(x2[0],y2[0]))
		
		# Draw intersection points onto the output image
		cv2.circle(img1_foe, (x1, y1), 5, (255, 0, 0), -1)

		# Get moving objects
		img1_moving_objects, moving_point_indices = draw_independently_moving_points_movingcam(img1_gray, lines1, pts1, pts2,  (x1, y1), (x2, y2))
	else:
		# Get moving objects
		img1_moving_objects, moving_point_indices = draw_independently_moving_points_stablecam(img1_gray, pts1, pts2)
	
	
	# Display image 1 with motion vectors and focus of expansion point
	plt.imshow(img1_foe)
	plt.show()
	
	# Display Moving Objects
	plt.imshow(img1_moving_objects)
	plt.show()
		
	# Get clusters and display the result
	clusters = cluster_moving_objects(img1_gray, moving_point_indices, pts1)
	plt.imshow(clusters)
	plt.show()
		
