import cv2

def detect_dark_traffic_lights(image):
    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 设置红色信号灯的颜色范围
    lower_red = (0, 100, 100)
    upper_red = (10, 255, 255)

    # 在图像中提取红色区域
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # 计算红色区域的像素总数
    red_pixels = cv2.countNonZero(red_mask)

    # 如果红色像素数量较低，则信号灯暗灭
    if red_pixels < 100:
        return True
    else:
        return False

# 读取测试图像
image = cv2.imread('traffic_light.jpg')

# 进行信号灯暗灭检测
is_dark = detect_dark_traffic_lights(image)

if is_dark:
    print("交通信号灯暗灭")
else:
    print("交通信号灯正常工作")
