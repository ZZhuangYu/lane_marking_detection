from collections import namedtuple

import cv2 as cv
import numpy as np


def draw_polynomial_curve(image, curve, color=(255, 0, 0), thickness=5):
    points = curve.get_points()
    num_of_points = len(points)
    for i in range(num_of_points - 1):
        start_u = points[i].u
        start_v = points[i].v
        end_u = points[i+1].u
        end_v = points[i+1].v
        image = cv.line(image, (start_u, start_v), (end_u, end_v), color, thickness)
    return image


def draw_drivable_area(image, left_curve, right_curve, color=(0, 255, 0)):
    left_points = left_curve.get_points()
    right_points = right_curve.get_points()
    assert (len(left_points) == len(right_points))
    num_of_points = len(left_points)
    for i in range(num_of_points - 1):
        left_start_u = left_points[i].u
        left_start_v = left_points[i].v
        left_end_u = left_points[i+1].u
        left_end_v = left_points[i+1].v

        right_start_u = right_points[i].u
        right_start_v = right_points[i].v
        right_end_u = right_points[i + 1].u
        right_end_v = right_points[i + 1].v

        drivable_area_points = np.array([[left_end_u, left_end_v], [right_end_u, right_end_v],
                                         [right_start_u, right_start_v], [left_start_u, left_start_v]])

        image = cv.fillConvexPoly(image, drivable_area_points, color)
    return image


def draw_lane_marking(image, pixels, color=(0, 255, 0), thickness=1):
    v_values = pixels[0]
    u_values = pixels[1]
    assert (len(v_values) == len(u_values))
    num_of_pixels = len(v_values)
    for i in range(num_of_pixels - 1):
        start_v = v_values[i]
        start_u = u_values[i]
        end_v = v_values[i + 1]
        end_u = u_values[i + 1]
        image = cv.line(image, (start_u, start_v), (end_u, end_v), color, thickness)
    return image


def remove_noise(bird_view, dilation_kernel=np.ones((17, 17)), erosion_kernel=np.ones((21, 21))):
    bird_view = cv.dilate(bird_view, dilation_kernel)
    bird_view = cv.erode(bird_view, erosion_kernel)
    return bird_view


def imshow(image, wait=0, title="test"):
    cv.imshow(title, image)
    cv.waitKey(wait)


Pixel = namedtuple("Pixel", ["u", "v"])