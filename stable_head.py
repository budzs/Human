import statistics
import dlib
import math
import pyrealsense2 as rs
import numpy as np
import cv2

config = rs.config()
try:
    rs.config.enable_device_from_file(config, "D:\\PycharmProjects\\pythonProject2\\files\\20211114_132016.bag",
                                      repeat_playback=False)
except (FileNotFoundError, IOError):
    print("Wrong file or file path")
pipeline = rs.pipeline()
profile = pipeline.start(config)
playback = profile.get_device().as_playback()
playback.set_real_time(False)

r_pupil = [200, 200]
l_pupil = [200, 200]

r_center = [200, 200]
l_center = [200, 200]
pic = 0
cv2.namedWindow("image")
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except (FileNotFoundError, IOError):
    print("Wrong file or file path")
right = [36, 37, 38, 39, 40, 41]
left = [42, 43, 44, 45, 46, 47]
nose = [27, 28, 29, 30]

kernel = np.ones((9, 9), np.uint8)

previous_ratio = 100
blink = False

f = open("result.txt", "w")


def shape_to_np(shape, dtype="int"):
    """
    Convert the 68 landmarks into a 2-tuple of (x, y)-coordinates
    :param shape: points from the detector and predictor
    :return: calculated coordinates
    """
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def eye_on_mask(mask, side):
    """
    Create mask on the eye
    :param mask: np.zeros(img.shape[:2], dtype=np.uint8)
    :param side: left or right
    :return: the created dark mask
    """
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def contouring(thresh, mid, right=False):
    """
    Left and right pupil location
    :param thresh: The created thresh in masking
    :param mid: calculated mid point
    :param right: side, True for right and False for left
    """
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
            global r_pupil
            r_pupil[0] = cx
            r_pupil[1] = cy
        else:
            global l_pupil
            l_pupil[0] = cx
            l_pupil[1] = cy
    except:
        pass


def nothing(x):
    """
    Trackbar help function
    :param x: -
    """
    pass


def midpoint(p1, p2):
    """
    Defining the mid-point between p1 and p2
    :param p1: start point
    :param p2: end point
    :return: calculated mid-point
    """
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def euclidean_distance(leftx, lefty, rightx, righty):
    """
    Defining the Euclidean distance
    :param leftx: first point's x coordinate
    :param lefty: first point's y coordinate
    :param rightx: second point's x coordinate
    :param righty: second point's y coordinate
    :return: calculated Euclidean distance
    """
    return np.sqrt((leftx - rightx) ** 2 + (lefty - righty) ** 2)


def get_EAR(side, facial_landmarks):
    """
    Defining the eye aspect ratio
    :param side: defines the current side (left, right)
    :param facial_landmarks: 68 facial landmark points
    :return: calculated EAR
    """
    if side == 0:
        eye_points = left
    elif side == 1:
        eye_points = right
    left_point = [facial_landmarks.part(eye_points[0]).x,
                  facial_landmarks.part(eye_points[0]).y]
    right_point = [facial_landmarks.part(eye_points[3]).x,
                   facial_landmarks.part(eye_points[3]).y]

    if side == 0:
        center_top = midpoint(facial_landmarks.part(eye_points[1]),
                              facial_landmarks.part(eye_points[2]))

        center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                                 facial_landmarks.part(eye_points[4]))
    elif side == 1:
        center_top = midpoint(facial_landmarks.part(eye_points[1]),
                              facial_landmarks.part(eye_points[2]))

        center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                                 facial_landmarks.part(eye_points[4]))

    hor_line_length = euclidean_distance(left_point[0], left_point[1],
                                         right_point[0], right_point[1])
    ver_line_length = euclidean_distance(center_top[0], center_top[1],
                                         center_bottom[0], center_bottom[1])
    # Calculating eye aspect ratio
    EAR = ver_line_length / hor_line_length
    if side == 1:
        global r_center
        r_center = ((left_point[0] + right_point[0]) / 2,
                    (left_point[1] + right_point[1]) / 2)
    elif side == 0:
        global l_center
        l_center = ((left_point[0] + right_point[0]) / 2,
                    (left_point[1] + right_point[1]) / 2)
    return EAR


def quarter(str1, str2):
    """
    Draw rectangle into gaze direction
    :param str1: Vertical direction
    :param str2: Horizontal direction
    """
    height, width, _ = img.shape

    if str1 == "UP":
        if str2 == "RIGHT":
            start = (0, 0)
            end = (int(width / 3), int(height / 3))
            cv2.rectangle(img, start, end, (255, 255, 0), 3)
        elif str2 == "LEFT":
            start = (int(2 * width / 3), 0)
            end = (width, int(height / 3))
            cv2.rectangle(img, start, end, (255, 255, 0), 3)
        elif str2 == "CENTER":
            start = (int(width / 3), 0)
            end = (int(2 * width / 3), int(height / 3))
            cv2.rectangle(img, start, end, (255, 255, 0), 3)

    elif str1 == "DOWN":
        if str2 == "RIGHT":
            start = (0, int(2 * height / 3))
            end = (int(width / 3), height)
            cv2.rectangle(img, start, end, (255, 255, 0), 3)
        elif str2 == "LEFT":
            start = (int(2 * width / 3), int(2 * height / 3))
            end = (width, height)
            cv2.rectangle(img, start, end, (255, 255, 0), 3)
        elif str2 == "CENTER":
            start = (int(width / 3), int(2 * height / 3))
            end = (int(2 * width / 3), height)
            cv2.rectangle(img, start, end, (255, 255, 0), 3)

    elif str1 == "CENTER":
        if str2 == "RIGHT":
            start = (0, int(height / 3))
            end = (int(width / 3), int(2 * height / 3))
            cv2.rectangle(img, start, end, (255, 255, 0), 3)
        elif str2 == "LEFT":
            start = (int(2 * width / 3), int(height / 3))
            end = (width, int(2 * height / 3))
            cv2.rectangle(img, start, end, (255, 255, 0), 3)
        elif str2 == "CENTER":
            start = (int(width / 3), int(height / 3))
            end = (int(2 * width / 3), int(2 * height / 3))
            cv2.rectangle(img, start, end, (255, 255, 0), 3)


def looking_at():
    """
    Gets gaze direction with stable head, writes in file, and call the drawing function
    Needs to be calibrated for the person
    """
    num_vertical = (r_pupil[0] - r_center[0]) + (l_pupil[0] - l_center[0]) / 2

    if num_vertical > 3:
        text_v = "LEFT"
    elif num_vertical < -7.0:
        text_v = "RIGHT"
    else:
        text_v = "CENTER"

    num_horizontal = (r_pupil[1] - (r_center[1] - 3.5)) + (l_pupil[1] - (l_center[1] - 3.5)) / 2

    if num_horizontal > 1.0:
        text_h = "DOWN"
    elif num_horizontal < -4.8:
        text_h = "UP"
    else:
        text_h = "CENTER"

    quarter(text_h, text_v)
    f.write("text_h: " + text_h + "text_v: " + text_v + "\n")


def automatic_thresh(picture):
    """
    Automatic threshold value calculation
    :param picture: picture
    :return: lower and upper threshold value
    """
    med = np.median(picture)
    sigma = 0.50
    lower_thresh = int(max(0, (1.0 - sigma) * med))
    upper_thresh = int(min(255, (1.0 + sigma) * med))
    return lower_thresh, upper_thresh


def masking(shape, img, auto):
    """
    Threshold value
    :param shape: points from the detector and predictor
    :param img: input image
    :param auto: threshold method True for auto, False for manual
    :return: created image and threshold
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = eye_on_mask(mask, left)
    mask = eye_on_mask(mask, right)
    mask = cv2.dilate(mask, kernel, 5)
    eyes = cv2.bitwise_and(img, img, mask=mask)
    mask = (eyes == [0, 0, 0]).all(axis=2)
    eyes[mask] = [255, 255, 255]
    picture = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
    mid = (shape[42][0] + shape[39][0]) // 2
    if auto:
        threshold_min, threshold_max = automatic_thresh(picture)
    else:
        threshold_min = cv2.getTrackbarPos('threshold', 'image')
        threshold_max = 255

    _, threshold = cv2.threshold(picture, threshold_min,
                                 threshold_max, cv2.THRESH_BINARY)
    threshold = cv2.erode(threshold, None, iterations=2)
    threshold = cv2.dilate(threshold, None, iterations=4)
    threshold = cv2.medianBlur(threshold, 3)
    threshold = cv2.bitwise_not(threshold)
    contouring(threshold[:, 0:mid], mid, img, True)
    contouring(threshold[:, mid:], mid, img)
    return img, threshold


cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
while True:
    blink = False
    frames = pipeline.wait_for_frames()
    playback.pause()
    depth_frame = frames.get_depth_frame()
    img = frames.get_color_frame()
    img = np.asanyarray(img.get_data())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = img
    rects = detector(gray)
    for rect in rects:
        shape = predictor(gray, rect)

        # Calculating eyes aspect ratio
        right_eye_ratio = get_EAR(1, shape)
        left_eye_ratio = get_EAR(0, shape)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio < 0.18:
            if previous_ratio >= 0.18:
                blink = True
                pic += 1
        previous_ratio = blinking_ratio

        if not blink:
            shape = shape_to_np(shape)
            img, thresh = masking(shape, img, True)
            looking_at()
            cv2.imwrite("files/20211114_132016/" + str(pic) + ".png", img)
            pic += 1

    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    playback.resume()
    if cv2.waitKey(1) == 27:
        break

f.close()
cv2.destroyAllWindows()
