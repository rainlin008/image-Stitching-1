# USAGE
# python stitch.py 

# import the necessary packages
from imageProcessing import ImageProcessing
import cv2
import glob

#read all the images inside a folder and sort by name
print("[INFO] loading images...")
image = [cv2.imread(file) for file in sorted(glob.glob("images/*"))]


imageProcesing = ImageProcessing()

#group the images by two
for i in range(0, len(image), 2):	

	print("[INFO] Step 1: Resize image for faster processing")
	imageA = imageProcesing.resize(image[i], 600)
	imageB = imageProcesing.resize(image[i+1], 600)

	imageProcesing.stitch([imageA, imageB])

	#reset variables for next iterartion
	imageProcesing.__init__()





	

	


