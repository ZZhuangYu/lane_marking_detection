import cv2 as cv
import numpy as np

from src.utils import Pixel, draw_polynomial_curve, imshow, draw_drivable_area
from src.polynomial_curve import PolynomialCurve


def detect_edges(image):
    """
    find edges in the given gray scale image
    :param image: gray scale image
    :return: binary image, where 1 is the edge, 0 is the background
    """
    height, width = image.shape
    edge_image = np.zeros((height, width))

    ############################################
    # TODO: implement your edge detection here #
    ############ your code starts ##############
    img_gau = cv.GaussianBlur(image, (9, 9), 0)
    #Canny
    edge_image = cv.Canny(img_gau,80,240)

    #Sobel
    #edge_image = cv.Sobel(img_gau, cv.CV_16S, 1, 1, ksize = 5)

    ############# your code ends ###############

    return edge_image


def perform_perspective_transform(front_view_image):
    """
    convert a front view image to bird view image by perspective transform
    :param front_view_image: front view image
    :return: bird view image, perspective transform matrix
    """
    height, width = front_view_image.shape

    top_left = [596, 428]
    top_right = [675, 428]
    bottom_right = [465, 555]
    bottom_left = [798, 555]

    top_left_dst = [465, 0]
    top_right_dst = [798, 0]
    bottom_right_dst = [465, height - 1]
    bottom_left_dst = [798, height - 1]

    bird_view_image_shape = (width, height)
    src = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst = np.float32([top_left_dst, top_right_dst, bottom_right_dst, bottom_left_dst])

    transform = cv.getPerspectiveTransform(src, dst)
    bird_view_image = cv.warpPerspective(front_view_image, transform, bird_view_image_shape)
    return bird_view_image, transform


def find_lane_marking_points(bird_view_image, u_mean, lane_marking_width=40):
    """
    find possible lane marking pixels around the mean
    :param bird_view_image:
    :param u_mean:
    :param lane_marking_width:
    :return: binary image
    """
    height, width = bird_view_image.shape
    lane_marking_points_image = np.zeros((height, width))
    left_u = u_mean - lane_marking_width * 0.5
    right_u = u_mean + lane_marking_width * 0.5
    for v in range(height):
        for u in range(width):
            lane_marking_points_image[v, u_mean] = 1.0
            if u >= left_u and u <= right_u and bird_view_image[v, u] > 0.0:
                lane_marking_points_image[v, u] = 1.0
    return lane_marking_points_image


def find_lane_marking_u_mean(bird_view_image, lane_width=200):
    """
    find u mean of left lane marking and right lane marking
    :param bird_view_image:
    :param lane_width:
    :return: left u, right u
    """
    height, width = bird_view_image.shape
    hits_map = np.sum(bird_view_image, axis=0)
    hits_index = [(u, hits_map[u]) for u in range(width)]
    sorted_hits_index = sorted(hits_index, key=lambda u_hits:u_hits[1], reverse=True)
    max_hits = sorted_hits_index[0][1]
    max_hits_u = sorted_hits_index[0][0]
    second_u = -1
    for i in range(1, width):
        u = sorted_hits_index[i][0]
        delta = np.abs(max_hits_u - u)
        if delta > lane_width:
            second_u = u
            break
    left_u = max_hits_u if max_hits_u < second_u else second_u
    right_u = max_hits_u if max_hits_u > second_u else second_u

    return left_u, right_u


def convert_points_to_front_view(perspective_transform, points_image):
    height, width = points_image.shape
    front_view_image = cv.warpPerspective(points_image, np.linalg.inv(perspective_transform), (width, height))
    front_view_points_array = np.where(front_view_image > 0.0)

    front_view_points = []
    for v, u in zip(front_view_points_array[0], front_view_points_array[1]):
        front_view_points.append(Pixel(u, v))

    return front_view_points


def process_image(filename):
    # load test image as BGR
    bgr_image = cv.imread(filename)

    # convert bgr image to hsv image
    hsv = cv.cvtColor(bgr_image, cv.COLOR_RGB2HSV)
    hsv_v = hsv[:, :, 2]

    # find edges
    edge_image = detect_edges(hsv_v)

    # convert front view edge image to bird view, to find possible lane marking points
    start_v = 555
    end_v = 428
    bird_view_image, transform = perform_perspective_transform(edge_image)

    # find u mean of left lane marking and right lane marking
    left_lane_marking_u, right_lane_marking_u = find_lane_marking_u_mean(bird_view_image)

    # lane marking points in bird view
    left_lane_marking_points = find_lane_marking_points(bird_view_image, left_lane_marking_u)
    right_lane_marking_points = find_lane_marking_points(bird_view_image, right_lane_marking_u)

    # transform lane marking points back to front view, with the inverse perspective transform matrix
    left_points_front_view = convert_points_to_front_view(transform, left_lane_marking_points)
    right_points_front_view = convert_points_to_front_view(transform, right_lane_marking_points)

    # three order polynomial curve fitting
    left_poly_curve = PolynomialCurve(start_v, end_v, left_points_front_view)
    right_poly_curve = PolynomialCurve(start_v, end_v, right_points_front_view)
    # draw polynomial curve on original image
    bgr_image = draw_polynomial_curve(bgr_image, left_poly_curve)
    bgr_image = draw_polynomial_curve(bgr_image, right_poly_curve)
    # draw drivable area
    bgr_image = draw_drivable_area(bgr_image, left_poly_curve, right_poly_curve)
    return bgr_image


if __name__ == "__main__":
    test_image_filename = "../data/test.png"
    lane_marking = process_image(test_image_filename)
    imshow(lane_marking)