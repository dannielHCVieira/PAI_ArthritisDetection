import cv2



def shape_selection(event, x, y, flags, param):
	# grab references to the global variables
	global ref_point, crop

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being performed
	if event == cv2.EVENT_LBUTTONDOWN:
		ref_point = [(x, y)]

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		ref_point.append((x, y))

		# draw a rectangle 
		cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

def create_environment(imageLoaded):
	# initialize the list of reference point
	global ref_point
	ref_point = []
	crop = False

	global image
	# read the image, clone it, and setup the mouse callback function
	image = cv2.imread("test.jpeg")
	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", shape_selection)


	# keep looping until the 'q' key is pressed
	while True:
		# display the image and wait for a keypress
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF

		# press 'r' to reset the window
		if key == ord("r"):
			image = clone.copy()

		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			break

	if len(ref_point) == 2:
		crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:
															ref_point[1][0]]
		cv2.imshow("crop_img", crop_img)
		cv2.waitKey(0)


