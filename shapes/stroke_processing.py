import numpy as np
import cv2
import math


class StrokeProcessor:

    def __init__(self):
        pass


    

    def resample(self, points, spacing=10):

        if len(points) < 2:
            return points

        resampled = [points[0]]
        accumulated = 0

        for i in range(1, len(points)):

            p1 = np.array(points[i-1])
            p2 = np.array(points[i])

            dist = np.linalg.norm(p2 - p1)

            accumulated += dist

            if accumulated >= spacing:
                resampled.append(tuple(p2))
                accumulated = 0

        return resampled


    

    def simplify(self, points):

        if len(points) < 5:
            return points

        contour = np.array(points, dtype=np.int32).reshape((-1,1,2))

        perimeter = cv2.arcLength(contour, False)

        epsilon = 0.05 * perimeter

        simplified = cv2.approxPolyDP(contour, epsilon, False)

        simplified_points = []

        for p in simplified:
            simplified_points.append((p[0][0], p[0][1]))

        return simplified_points


    

    def angle(self, p1, p2, p3):

        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        ba = a - b
        bc = c - b

        if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
            return 180

        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return math.degrees(math.acos(cos_angle))


    

    def detect_corners(self, points):

        corners = []

        threshold = 60

        for i in range(2, len(points)):

            p1 = points[i-2]
            p2 = points[i-1]
            p3 = points[i]

            ang = self.angle(p1, p2, p3)

            if ang < threshold:
                corners.append(p2)

        return corners


    

    def merge_corners(self, corners, threshold=25):

        if not corners:
            return corners

        merged = [corners[0]]

        for c in corners[1:]:

            prev = merged[-1]

            dist = np.linalg.norm(np.array(c) - np.array(prev))

            if dist > threshold:
                merged.append(c)

        return merged


    

    def is_circle(self, contour):

        contour = np.array(contour, dtype=np.int32).reshape((-1,1,2))

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return False

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        return circularity > 0.7


    

    def classify(self, stroke):

        if len(stroke) < 10:
            return None, []

        resampled = self.resample(stroke)

        simplified = self.simplify(resampled)

        corners = self.detect_corners(simplified)

        corners = self.merge_corners(corners)

        corner_count = len(corners)

        if self.is_circle(resampled):
            return "CIRCLE", corners

        if corner_count == 2:
            return "LINE", corners

        if corner_count == 3:
            return "TRIANGLE", corners

        if corner_count == 4:
            return "RECTANGLE", corners

        if corner_count == 5:
            return "PENTAGON", corners

        if corner_count == 6:
            return "HEXAGON", corners

        return f"{corner_count}-GON", corners