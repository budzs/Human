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
f = open("131047.txt", "w")
r_pupil = [200, 200]
l_pupil = [200, 200]

r_center = [200, 200]
l_center = [200, 200]
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

counter = 0
memory_yaw = 0
memory_pitch = 0
memory_roll = 0
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


def get_angle(bottom, top):
    """
    Computes the angle
    :param bottom: start point
    :param top: end point
    :return: calculated angle in degrees
    """
    dx = bottom[0] - top[0]
    dy = bottom[1] - top[1]
    return np.degrees(np.arctan2(dy, dx))


def align_img(img, nose_center, angle):
    """
    Rotate the image to check the roll angle
    :param img: input image
    :param nose_center: center of the rotation
    :param angle: angle of the rotation
    :return: rotated image
    """
    height, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center=nose_center, angle=angle, scale=1)
    img = cv2.warpAffine(img, M, (width, height))
    return img


def depth_threshold(shape):
    """
    Calculate depth threshold for the whole picture
    :param shape: points from the detector and predictor
    :return: threshold value
    """
    values = []
    for i in shape:
        depth = depth_frame[i[1], i[0]]
        if 0 < depth < 2000:
            values.append(depth)
    med = statistics.median(values)
    dev = statistics.pstdev(values)
    return med + dev


def mean(point):
    """
    Noise filtering by calculating mean of the neighbours
    :param point: needed point
    :return: calculated mean
    """
    neighbours = [(point[0] - 1, point[1] - 1), (point[0] - 1, point[1]), (point[0] - 1, point[1] + 1),
                  (point[0], point[1] - 1), (point[0], point[1]), (point[0], point[1] + 1),
                  (point[0] + 1, point[1] - 1), (point[0] + 1, point[1]), (point[0] + 1, point[1] + 1)]
    mean_depth = 0
    count = 0
    nums = []
    for i in range(len(neighbours)):
        if neighbours[i][0] < 480 and neighbours[i][1] < 640:
            value = depth_image[neighbours[i][0], neighbours[i][1]]
        else:
            value = 0
        if value != 0 and value < depth_threshold(shape):
            mean_depth += value
            nums.append(value)
            count += 1

    if count != 0 and mean_depth != 0:
        return mean_depth / count
    else:
        try:
            mean_val = depth_image[point[0], point[1]]
            if mean_val > depth_threshold(shape):
                return 0
            return mean_val
        except:
            return 0


def yaw_angle():
    """
    Calculate Yaw angle
    :return: angle in degrees
    """
    r_center = (int((shape[36][0] + shape[39][0]) / 2), int((shape[36][1] + shape[39][1]) / 2))
    l_center = (int((shape[42][0] + shape[45][0]) / 2), int((shape[42][1] + shape[45][1]) / 2))
    d = l_center[0] - r_center[0]
    e1 = mean(shape[45])
    e2 = mean(shape[36])
    m = e2 - e1
    angle = math.atan2(m, d)
    # offset = m / d * e1
    return math.degrees(angle)


def pitch_angle():
    """
    Calculate Pitch angle
    :return: angle in degrees
    """
    forehead = (int((shape[20][0] + shape[23][0]) / 2), int((shape[20][1] + shape[23][1]) / 2))
    chin = (int((shape[6][0] + shape[10][0]) / 2), int((shape[6][1] + shape[10][1]) / 2))

    depth_forehead = mean(forehead)
    depth_chin = mean(chin)

    try:
        angle = math.atan2((depth_chin - depth_forehead), (chin[1] - forehead[1]))
        return math.degrees(angle)
    except:
        return 0


def roll_angle():
    """
    Calculate Roll angle
    :return: angle in degrees
    """
    try:
        angle = get_angle(l_center, r_center)
        return angle
    except:
        return 0


cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
while True:
    blink = False
    frames = pipeline.wait_for_frames()
    playback.pause()
    depth_frame = frames.get_depth_frame()
    img = frames.get_color_frame()
    if not depth_frame:
        continue
    depth_image = np.asanyarray(depth_frame.get_data())
    img = np.asanyarray(img.get_data())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    counter += 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = img
    rects = detector(gray)

    if len(rects) == 0:
        s = "%s,%s,%s\n" % (0, 0, 0)
        f.write(s)
    for rect in rects:
        shape = predictor(gray, rect)

        # Calculating eyes aspect ratio
        right_eye_ratio = get_EAR(1, shape)
        left_eye_ratio = get_EAR(0, shape)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio < 0.18:
            if previous_ratio >= 0.18:
                blink = True
        previous_ratio = blinking_ratio

        if not blink:
            shape = shape_to_np(shape)
            img, thresh = masking(shape, img)
            looking_at()

            pitch = round(pitch_angle(), 4)
            if pitch > 180:
                pitch = 0
            if counter < 4 and (pitch == 0 or abs(pitch - memory_pitch) > 10 ):
                pitch = memory_pitch
            else:
                memory_pitch = pitch

            yaw = round(yaw_angle(), 4)
            if yaw > 180:
                yaw = 0
            if counter < 4 and (yaw == 0 or abs(yaw - memory_yaw) > 10):
                yaw = memory_yaw
            else:
                memory_yaw = yaw

            roll = round(roll_angle(), 4)
            if roll > 180:
                roll = 0
            if counter < 4 and (roll == 0 or abs(roll - memory_roll) > 10):
                roll = memory_roll
            else:
                memory_roll = roll
            if counter >= 4:
                counter = 0
            # cv2.putText(img, ('pitch:{}  , yaw:{}  , roll:{}'.format(pitch, yaw, roll)), (100, 100),
            #            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            s = "%s,%s,%s\n" % (pitch, yaw, roll)
            f.write(s)
    # cv2.imshow('eyes', img)
    # cv2.imshow("image", thresh)
    playback.resume()

    if cv2.waitKey(1) == 27:
        break
f.close()
cv2.destroyAllWindows()
