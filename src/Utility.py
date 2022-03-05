import os
import cv2
import numpy as np


def get_files(directory_path) -> list:
    files = os.listdir(directory_path)
    return [f'{directory_path}/{file_name}' for file_name in files]


def get_images(files: list) -> dict:
    image_lookup = dict()
    for file in files:
        image = cv2.imread(file)
        image_lookup[file] = image
    return image_lookup


def get_hough_circles(image) -> tuple:
    inner_circle = None
    outer_circle = None
    cv2.GaussianBlur(image, (7, 7), 4, dst=image)
    inner_circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 500, param1=130, param2=26, minRadius=22,
                                     maxRadius=125)
    if inner_circles is not None:
        inner_circles = np.round(inner_circles[0, :]).astype('int')
        inner_circle = inner_circles[0]

    outer_circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 500, param1=30, param2=22, minRadius=65,
                                     maxRadius=115)
    if outer_circles is not None:
        outer_circles = np.round(outer_circles[0, :]).astype('int')
        outer_circle = outer_circles[0]

    return inner_circle, outer_circle


def draw_circles(image, circle):
    x, y, r = circle
    cv2.circle(image, (x, y), r, (0, 255, 0), thickness=2)
    cv2.circle(image, (x, y), 2, (0, 0, 255), thickness=-1)


def circle_extraction(image, inner_circle, outer_circle):
    center_x1, center_y1, r1 = inner_circle
    center_x2, center_y2, r2 = outer_circle
    h, w = image.shape
    for x in range(w):
        for y in range(h):
            sum_squares1 = calculate_circle(x, y, center_x1, center_y1)
            r_squared1 = r1**2
            r_squared2 = r2**2
            # used same center for both circles so the iris left would be same width all the way around
            if sum_squares1 <= r_squared1:
                image[y, x] = 0
            elif sum_squares1 > r_squared2:
                image[y, x] = 0


def calculate_circle(x, y, center_x, center_y) -> int:
    return ((x - center_x)**2) + ((y - center_y)**2)


def snakes_eyelid_extraction(binary_image, contours_image, path):
    keep_going = True
    count = 0
    while keep_going:
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(binary_image, contours, -1, 255, thickness=-1, lineType=cv2.LINE_AA)
        cv2.drawContours(contours_image, contours, -1, 0, thickness=-1, lineType=cv2.LINE_AA)
        if count % 2 == 0:
            display_image(contours_image, f'{path}: Iris Contours')
            print('Type False to stop finding contours')
            val = input('Keep going: ').upper()
            if val == 'FALSE':
                keep_going = False
        count = count + 1


def convert_to_polar_image(image):
    array = np.zeros(60000, dtype=np.uint8)
    index = 0
    h, w = image.shape
    for x in range(w):
        for y in range(h):
            if image[y, x] != 0:
                array[index] = image[y, x]
                index = index + 1
    new_image = np.reshape(array, (150, 400))
    return new_image


def display_image(image, image_name: str):
    cv2.namedWindow(image_name)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(image_name)
