All 40 Experiments — Code Only (OpenCV Python)

Below are compact, runnable Python scripts (one per experiment) using OpenCV (cv2) and numpy where needed. Save each snippet as expXX_name.py or run in a cell. Replace file names (img.jpg, video.mp4, etc.) with your actual paths.


---

Experiment 01 — Read Image & Convert to Grayscale

import cv2

img = cv2.imread('img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', img)
cv2.imshow('Gray', gray)
cv2.imwrite('exp01_gray.jpg', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 02 — Gaussian Blur

import cv2

img = cv2.imread('img.jpg')
blur = cv2.GaussianBlur(img, (15, 15), 0)
cv2.imshow('Blur', blur)
cv2.imwrite('exp02_blur.jpg', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 03 — Canny Edge Detection

import cv2

img = cv2.imread('img.jpg', 0)
edges = cv2.Canny(img, 100, 200)
cv2.imshow('Edges', edges)
cv2.imwrite('exp03_edges.jpg', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 04 — Dilation

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('Dilated', dilated)
cv2.imwrite('exp04_dilated.jpg', dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 05 — Erosion

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.ones((5,5), np.uint8)
eroded = cv2.erode(img, kernel, iterations=1)
cv2.imshow('Eroded', eroded)
cv2.imwrite('exp05_eroded.jpg', eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 06 — Video Read & Speed Control (Normal/Slow/Fast)

import cv2

cap = cv2.VideoCapture('video.mp4')
delay = 40  # normal

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video', frame)
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # slow
        delay = 120
    elif key == ord('n'):  # normal
        delay = 40
    elif key == ord('f'):  # fast
        delay = 10

cap.release()
cv2.destroyAllWindows()

Experiment 07 — Webcam Capture

import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

Experiment 08 — Image Scaling (Resize)

import cv2

img = cv2.imread('img.jpg')
big = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imshow('Big', big)
cv2.imshow('Small', small)
cv2.imwrite('exp08_big.jpg', big)
cv2.imwrite('exp08_small.jpg', small)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 09 — Rotation (Affine)

import cv2
import numpy as np

img = cv2.imread('img.jpg')
(h,w) = img.shape[:2]
M = cv2.getRotationMatrix2D((w/2,h/2), 45, 1.0)  # rotate 45 degrees
rot = cv2.warpAffine(img, M, (w, h))
cv2.imshow('Rotated', rot)
cv2.imwrite('exp09_rotated.jpg', rot)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 10 — Translation (Shift)

import cv2
import numpy as np

img = cv2.imread('img.jpg')
M = np.float32([[1, 0, 50], [0, 1, 100]])
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('Shifted', shifted)
cv2.imwrite('exp10_shifted.jpg', shifted)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 11 — Affine Transformation (3 points)

import cv2
import numpy as np

img = cv2.imread('img.jpg')
rows, cols = img.shape[:2]
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('Affine', dst)
cv2.imwrite('exp11_affine.jpg', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 12 — Perspective Transformation (Image)

import cv2
import numpy as np

img = cv2.imread('img.jpg')
rows, cols = img.shape[:2]
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1, pts2)
warp = cv2.warpPerspective(img, M, (300,300))
cv2.imshow('Perspective', warp)
cv2.imwrite('exp12_perspective.jpg', warp)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 13 — Perspective Transformation (Video)

import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1, pts2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    warp = cv2.warpPerspective(frame, M, (300,300))
    cv2.imshow('Warped Video', warp)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

Experiment 14 — Homography (feature-match + warp) — skeleton

import cv2
import numpy as np

img1 = cv2.imread('img1.jpg')  # source
img2 = cv2.imread('img2.jpg')  # destination

# Use ORB for matching (example). In practice, tune & check matches.
orb = cv2.ORB_create()
k1, d1 = orb.detectAndCompute(img1, None)
k2, d2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(d1, d2)
matches = sorted(matches, key=lambda x: x.distance)[:50]

pts_src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
pts_dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
warped = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
cv2.imwrite('exp14_homography.jpg', warped)

Experiment 15 — Direct Linear Transformation (DLT) Outline (concept code)

import numpy as np

# DLT requires at least 4 correspondences pts_src, pts_dst as (N,2)
# This is a conceptual skeleton — for real use prefer cv2.findHomography
def dlt_homography(pts_src, pts_dst):
    N = pts_src.shape[0]
    A = []
    for i in range(N):
        x,y = pts_src[i]
        u,v = pts_dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    A = np.array(A)
    U,S,Vt = np.linalg.svd(A)
    h = Vt[-1,:]
    H = h.reshape(3,3)
    return H/H[2,2]

Experiment 16 — Canny (tuned)

import cv2

img = cv2.imread('img.jpg', 0)
edges = cv2.Canny(img, 50, 150)  # tune thresholds
cv2.imshow('Canny Tuned', edges)
cv2.imwrite('exp16_canny_tuned.jpg', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 17 — Sobel X

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
abs_sobelx = np.absolute(sobelx)
sobelx_8u = np.uint8(abs_sobelx)
cv2.imshow('Sobel X', sobelx_8u)
cv2.imwrite('exp17_sobelx.jpg', sobelx_8u)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 18 — Sobel Y

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
abs_sobely = np.absolute(sobely)
sobely_8u = np.uint8(abs_sobely)
cv2.imshow('Sobel Y', sobely_8u)
cv2.imwrite('exp18_sobely.jpg', sobely_8u)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 19 — Sobel XY (magnitude)

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(sobelx*2 + sobely*2)
mag = np.uint8(np.clip(mag, 0, 255))
cv2.imshow('Sobel XY', mag)
cv2.imwrite('exp19_sobelxy.jpg', mag)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 20 — Laplacian Sharpening (negative center)

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
sharpen = cv2.filter2D(img, -1, kernel)
cv2.imshow('Sharpen', sharpen)
cv2.imwrite('exp20_sharpen.jpg', sharpen)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 21 — Laplacian Sharpening (diagonal extension)

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharpen = cv2.filter2D(img, -1, kernel)
cv2.imshow('Diag Sharpen', sharpen)
cv2.imwrite('exp21_sharpen_diag.jpg', sharpen)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 22 — Laplacian (positive center)

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.array([[1,1,1],[1,-7,1],[1,1,1]])
res = cv2.filter2D(img, -1, kernel)
cv2.imshow('Pos Center Lap', res)
cv2.imwrite('exp22_poslap.jpg', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 23 — Unsharp Masking

import cv2

img = cv2.imread('img.jpg')
gauss = cv2.GaussianBlur(img, (9,9), 10.0)
unsharp = cv2.addWeighted(img, 1.5, gauss, -0.5, 0)
cv2.imshow('Unsharp', unsharp)
cv2.imwrite('exp23_unsharp.jpg', unsharp)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 24 — High-Boost Filtering

import cv2

img = cv2.imread('img.jpg')
A = 1.5
blur = cv2.GaussianBlur(img, (9,9), 10)
high_boost = cv2.addWeighted(img, A, blur, (1-A), 0)
cv2.imshow('High Boost', high_boost)
cv2.imwrite('exp24_highboost.jpg', high_boost)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 25 — Gradient Masking (simple)

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
grad = cv2.filter2D(img, -1, kernelx)
cv2.imshow('Gradient Mask', grad)
cv2.imwrite('exp25_gradient.jpg', grad)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 26 — Watermarking (text with alpha)

import cv2

img = cv2.imread('img.jpg')
(h,w) = img.shape[:2]
overlay = img.copy()
cv2.putText(overlay, 'Watermark', (w-220,h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
alpha = 0.4
watermarked = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
cv2.imshow('Watermarked', watermarked)
cv2.imwrite('exp26_watermark.jpg', watermarked)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 27 — Cropping, Copy & Paste ROI

import cv2

img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')
roi = img1[50:200, 100:300]  # change coords as needed
img2[10:10+roi.shape[0], 10:10+roi.shape[1]] = roi
cv2.imshow('Pasted', img2)
cv2.imwrite('exp27_paste.jpg', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 28 — Boundary Detection (Convolution)

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
boundary = cv2.filter2D(img, -1, kernel)
cv2.imshow('Boundary', boundary)
cv2.imwrite('exp28_boundary.jpg', boundary)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 29 — Morphology: Erosion

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.ones((3,3), np.uint8)
eroded = cv2.erode(img, kernel, iterations=1)
cv2.imshow('Erosion', eroded)
cv2.imwrite('exp29_erosion.jpg', eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 30 — Morphology: Dilation

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('Dilation', dilated)
cv2.imwrite('exp30_dilation.jpg', dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 31 — Morphology: Opening

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening', opening)
cv2.imwrite('exp31_opening.jpg', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 32 — Morphology: Closing

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing', closing)
cv2.imwrite('exp32_closing.jpg', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 33 — Morphology: Gradient

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.ones((5,5), np.uint8)
grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('Morph Gradient', grad)
cv2.imwrite('exp33_gradient.jpg', grad)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 34 — Morphology: Top Hat

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.ones((15,15), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('Top Hat', tophat)
cv2.imwrite('exp34_tophat.jpg', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 35 — Morphology: Black Hat

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
kernel = np.ones((15,15), np.uint8)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('Black Hat', blackhat)
cv2.imwrite('exp35_blackhat.jpg', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 36 — Watch Detection (Template Matching)

import cv2

img = cv2.imread('scene.jpg')
template = cv2.imread('watch_template.jpg')
th, tw = template.shape[:2]
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
if max_val > 0.7:
    top_left = max_loc
    bottom_right = (top_left[0]+tw, top_left[1]+th)
    cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)
cv2.imshow('Template Match', img)
cv2.imwrite('exp36_watch_detect.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 37 — Video Reverse Play

import cv2

cap = cv2.VideoCapture('video.mp4')
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

for frame in reversed(frames):
    cv2.imshow('Reverse Play', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

Experiment 38 — Face Detection (Haar Cascade)

import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('group.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
cv2.imshow('Faces', img)
cv2.imwrite('exp38_faces.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

Experiment 39 — Vehicle Detection (Background Subtraction)

import cv2

cap = cv2.VideoCapture('traffic.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow('Vehicles', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

Experiment 40 — Draw Rectangle & Extract Object (ROI)

import cv2

img = cv2.imread('img.jpg')
x,y,w,h = 50,50,200,150  # change as needed or use mouse callback to select
cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
obj = img[y:y+h, x:x+w]
cv2.imshow('Object', obj)
cv2.imwrite('exp40_object.jpg', obj)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


---

If you’d like any of the following, tell me which one and I’ll do it now:

Save all snippets into one ZIP with individual .py files and provide the download link.

Convert these into a single multi-experiment Python script with a menu to select which experiment to run.

Add inline comments / explanations for each line of code.

Provide accompanying sample images or expected output screenshots inserted into the DOC (I can generate placeholders).
