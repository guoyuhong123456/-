import numpy as np
import cv2


def order_points(pts):
    rsct = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axi=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax[s]]
    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2)+((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0])**2)+((tr[1]-tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
    heightB = np.sort(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight - 1],
        [0, maxHeight-1]], dtype="float32")
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 读取输入
image = cv2.imread('receipt.jpg')
ratio = image.shape[0]/500.0
org = image.copy()
image = resize(org, height=500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# 展示预处理结果
print("STEP 1:边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
# 遍历轮廓
for c in cnts:
    peri = cv2.arclength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break
# 展示结果
print("STEP 2:获得轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 透视变换
warped = four_point_transform(org, screenCnt.reshape(4, 2)*ratio)

# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GR)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg', ref)

# 展示结果
print("STEP 3:变换")
cv2.imshow("Original", resize(orig, ieight=650))
cv2.imshow("Scanned", resize(ref, height=650))
cv2.waitKey(0)



