import Utility as Util
import cv2
import numpy as np


def main():
    resources_dir = 'Resources'
    image_paths = Util.get_files(resources_dir)
    image_lookup = Util.get_images(image_paths)
    for path in image_paths:
        image = image_lookup[path]
        Util.display_image(image, f'{path}: Original')

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inner_circle, outer_circle = Util.get_hough_circles(grayscale)
        if inner_circle is None or outer_circle is None:
            continue
        image_circles = image.copy()
        Util.draw_circles(image_circles, inner_circle)
        Util.draw_circles(image_circles, outer_circle)
        Util.display_image(image_circles, f'{path}: Added Circles')

        iris_image = image.copy()
        iris_image = cv2.cvtColor(iris_image, cv2.COLOR_BGR2GRAY)
        Util.circle_extraction(iris_image, inner_circle, outer_circle)
        Util.display_image(iris_image, f'{path}: Extracted Iris')

        binary_image = iris_image.copy()
        cv2.GaussianBlur(binary_image, (7, 7), 2, dst=binary_image)
        ret, binary_image = cv2.threshold(binary_image, 128, 255, cv2.THRESH_BINARY)
        Util.display_image(binary_image, f'{path}: Binary Image')

        contours_image = iris_image.copy()
        Util.snakes_eyelid_extraction(binary_image, contours_image, path)
        Util.display_image(contours_image, f'{path}: Final Contours')

        unwrapped_iris = contours_image.copy()
        unwrapped_iris = Util.convert_to_polar_image(unwrapped_iris)
        Util.display_image(unwrapped_iris, f'{path}: Unwrapped Iris')


if __name__ == '__main__':
    main()
