import numpy as np
import cv2


class SketchPanel:

    def __init__(self, width=640, height=480):

        self.width  = width
        self.height = height

        self.stroke_layer      = np.zeros((height, width, 3), dtype=np.uint8)
        self.background        = np.zeros((height, width, 3), dtype=np.uint8)
        self.canvas            = np.zeros((height, width, 3), dtype=np.uint8)

        self.active            = False
        self.prev_point        = None
        self.current_stroke    = []
        self.completed_strokes = []

        self.MIN_DRAW_DIST = 6   # pixels — below this nothing is drawn

        self.draw_grid()


    def draw_grid(self):

        self.background[:] = (38, 16, 16)

        small_gap = 15
        large_gap = 75

        for x in range(0, self.width, small_gap):
            cv2.line(self.background, (x, 0), (x, self.height),
                     (60, 75, 95), 1)
        for y in range(0, self.height, small_gap):
            cv2.line(self.background, (0, y), (self.width, y),
                     (60, 75, 95), 1)

        for x in range(0, self.width, large_gap):
            cv2.line(self.background, (x, 0), (x, self.height),
                     (90, 110, 135), 1)
        for y in range(0, self.height, large_gap):
            cv2.line(self.background, (0, y), (self.width, y),
                     (90, 110, 135), 1)

        cross_size = 4
        for x in range(0, self.width, large_gap):
            for y in range(0, self.height, large_gap):
                cv2.line(self.background,
                         (x - cross_size, y), (x + cross_size, y),
                         (140, 165, 190), 1)
                cv2.line(self.background,
                         (x, y - cross_size), (x, y + cross_size),
                         (140, 165, 190), 1)


    def draw_stroke(self, point):

        if self.prev_point is None:
            self.prev_point = point
            self.current_stroke.append(point)
            return

        dx   = point[0] - self.prev_point[0]
        dy   = point[1] - self.prev_point[1]
        dist = (dx*dx + dy*dy) ** 0.5

        # skip tiny movements — prevents jitter dots and micro lines
        if dist < self.MIN_DRAW_DIST:
            return

        # Better smoothing: 70% previous, 30% current for smoother line with continuity
        smooth_x = int(self.prev_point[0] * 0.70 + point[0] * 0.30)
        smooth_y = int(self.prev_point[1] * 0.70 + point[1] * 0.30)
        smooth_point = (smooth_x, smooth_y)

        cv2.line(self.stroke_layer, self.prev_point,
                 smooth_point, (255, 255, 255), 2)

        self.prev_point = smooth_point
        self.current_stroke.append(smooth_point)


    def finish_stroke(self):

        if len(self.current_stroke) > 2:
            self.completed_strokes.append(self.current_stroke)

        self.current_stroke = []
        self.prev_point     = None


    def render(self, frame):

        self.canvas[:] = (38, 16, 16)

        if self.active:
            self.canvas[:] = self.background
            cv2.add(self.background, self.stroke_layer, self.canvas)

        return self.canvas