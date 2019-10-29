The goal of this project is to build a Computer Vision System that stitch two camera views to generate
a panorama. 

The algorithm can be described as follows:<br>
• Step 0: Read all the images from the images folder sorting by name<br>
• For every two images:<br>
  o Step 1: Resize images for faster processing<br>
  o Step 2: Detect keypoints and extract local invariant descriptors from the two images<br>
  o Step 3: Match the descriptors between the two images<br>
  o Step 4: Use RANSAC algorithm to estimate a Homography Matrix using our matched feature vectors<br>
  o Step 5: Apply warping transformation using the Homography Matrix obtained from Step 4<br>
  o Step 6: Visualize keypoint correspondences between two images<br>
  o Step 7: Plot the images in only one window<br>
  o Step 8: Remove/Reduce black edges from the Panorama if exists<br>
  
  
On the two samples bellow, the first one needs to apply step 8, while on the second one the panorama
is perfect, so we don’t need to apply step 8. <br>
<img src="samples/Figure_2.png" width="1000px" >



<b>RUN:</b> type python stitch.py in the command line
