import numpy as np
import cv2


class ContourProcessor:

    def simplify_stroke(self, stroke):

        if len(stroke) < 5:
            return None

        contour = np.array(stroke, dtype=np.int32)
        contour = contour.reshape((-1,1,2))

        perimeter = cv2.arcLength(contour, False)

        epsilon = 0.02 * perimeter

        simplified = cv2.approxPolyDP(contour, epsilon, False)

        simplified_points = []

        for p in simplified:
            simplified_points.append((p[0][0], p[0][1]))

        return simplified_points