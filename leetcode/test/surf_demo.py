import cv2

img = cv2.imread("feature.jpg")
img = cv2.resize(img, (136 * 3, 76 * 3))
cv2.imshow("original", img)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 使用SIFT
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray, None)

cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                  color=(51, 163, 236))
cv2.imshow("SIFT", img)

# 使用SURF
img = cv2.imread("feature.jpg")
img = cv2.resize(img, (136 * 3, 76 * 3))

surf = cv2.xfeatures2d.SURF_create()
keypoints, descriptor = surf.detectAndCompute(gray, None)

cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                  color=(51, 163, 236))
cv2.imshow("SURF", img)

img = cv2.imread("feature.jpg")
img = cv2.resize(img, (136 * 3, 76 * 3))

cv2.waitKey(0)
cv2.destroyAllWindows()
