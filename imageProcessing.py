import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageProcessing:

	def __init__(self):
		self.rawMatches = None
		self.homograpyTuple = None
		self.result = None
		self.matches = None
		self.status = None


	#Step 1: Resize image for faster processing
	def resize(self, image, width=None, inter=cv2.INTER_AREA):

	    	# initialize the dimensions of the image to be resized and grab the image size
		dim = None
		(h, w) = image.shape[:2]

	    	# if width is None, then return the original image
		if width is None: 
			return image

	    	# calculate the ratio of the width and construct the dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	    	# resize the image
		resized = cv2.resize(image, dim, interpolation=inter)

		return resized


	#Step 2: Detect keypoints and extract local invariant descriptors from the two input images.
	def detectAndDescribe(self, image):

		print("[INFO] Step 2: Detect keypoints and extract local invariant descriptors")

		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# detect and extract features from the image
		descriptor = cv2.xfeatures2d.SIFT_create()
		(kps, features) = descriptor.detectAndCompute(image, None)

		# convert the keypoints from KeyPoint objects to NumPy arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)


	#Step 3: Match the descriptors between the two images
	def matchKeypoints(self, featuresA, featuresB, kBestMatches = 2):

		print("[INFO] Step 3: Match the descriptors between the two images")

		# compute the raw matches and initialize the list of actual matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")

		#Step 3: Match the descriptors between the two images
		#Finds the k best matches for each descriptor from a query set 
		self.rawMatches = matcher.knnMatch(featuresA, featuresB, kBestMatches)
		
		return self.rawMatches


	#Step 4: Use the RANSAC algorithm to estimate a homography matrix using our matched feature vectors
	def homographyEstimation(self, kpsA, kpsB, ratio, reprojThresh, homographyThresh = 4, kBestMatches = 2):

		print("[INFO] Step 4: Estimate a homography matrix")

		matches = []

		# loop over the raw matches
		for m in self.rawMatches:
			# ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
			if len(m) == kBestMatches and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))


		# computing a homography requires at least 4 matches
		if len(matches) > homographyThresh:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)

			# return the matches along with the homograpy matrix and status of each matched point
			self.homograpyTuple = (matches, H, status)
			return self.homograpyTuple

		else:
			# otherwise, no homograpy could be computed
			print("[INFO] Step 4: XXXXXXXXXXX No match exit XXXXXXXXX")
			return self.homograpyTuple


	#Step 5: Apply a warping transformation using the homography matrix obtained from Step #4
	def warpingTransformation(self, featuresMatch, imageA, imageB):

		print("[INFO] Step 5: Apply warping transformation")

		# apply a perspective warp to stitch the images together
		(self.matches, H, self.status) = featuresMatch 
		self.result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		self.result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB


	#Step 6: Visualize keypoint correspondences between two images
	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status, drawKeypointThresh = 1):

		print("[INFO] Step 6: Draw keypoint correspondences")

		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully matched
			if s == drawKeypointThresh:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		return vis


	#Step 7: Plot the images in only one window
	def visualizeSameWindow(self, imageA, imageB, drawMatches):

		print("[INFO] Step 7: Plot the images in only one window")

		imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
		imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)
		result = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)

		sameWindow = {"Image A" : np.array(imageA), "Image B" : np.array(imageB), "Panorama" : np.array(result)}

		#Step 8: remove/reduce black edges from all image borders
		print("[INFO] Step 8: Remove/Reduce black edges from all image borders")
		resultcropped = self.trim(result)

		#if there are black edges show cropped image
		if (np.array_equal(resultcropped, result) == False):
			sameWindow.update( {"Panorama cropped" : np.array(resultcropped)} )

		if drawMatches is not None:
			drawMatches = cv2.cvtColor(drawMatches, cv2.COLOR_BGR2RGB)
			sameWindow.update( {"Matches between Image A and B" : np.array(drawMatches)} )

		plt.figure(figsize=(64, 64))

		# plot every image
		for i, (key, value) in enumerate(sameWindow.items()): 
			plt.subplot(2,3,i+1)
			plt.title(key)
			plt.axis('off')
			plt.imshow(value)


		plt.show()


	#Step 8: remove/reduce black edges from all image borders
	def trim(self, frame):

    		#crop top
		if not np.sum(frame[0]):
        		return self.trim(frame[1:])
    		#crop top
		if not np.sum(frame[-1]):
        		return self.trim(frame[:-2])
    		#crop top
		if not np.sum(frame[:,0]):
        		return self.trim(frame[:,1:])
    		#crop top
		if not np.sum(frame[:,-1]):
        		return self.trim(frame[:,:-2])
		return frame



	#Step 9: PUT EVERYTHING TOGETHER
	def stitch(self, images, ratio = 0.75, reprojThresh = 4.0, showMatches = True):

		# unpack the images, then detect keypoints and extract local invariant descriptors from them
		(imageB, imageA) = images

		#Step 2: Detect keypoints and extract local invariant descriptors from the two input images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)

		#Step 3: match features between the two images
		self.matchKeypoints(featuresA, featuresB)

		#Step 4: Use the RANSAC to estimate a homography matrix using matched feature vectors
		featuresMatch = self.homographyEstimation(kpsA, kpsB, ratio, reprojThresh)

		# if the match is None, then there aren't enough matched keypoints to create a panorama
		if featuresMatch  is None:
			return None

		#Step 5: Apply a warping transformation using the homography matrix obtained from Step #4
		self.warpingTransformation(featuresMatch, imageA, imageB)

		#Step 6: check to see if the keypoint matches should be visualized
		if showMatches:

			drawMatches = self.drawMatches(imageA, imageB, kpsA, kpsB, self.matches, self.status)

			#Step 7: Plot the images in only one window
			self.visualizeSameWindow(imageA, imageB, drawMatches)

		else:
			#Step 7: Plot the images in only one window
			self.visualizeSameWindow(imageA, imageB, None)


	

	

