import numpy as np
import cv2


class SketchPanel:

    def __init__(self, width=1280, height=720):

        self.width = width
        self.height = height
        self.current_stroke = []
        self.completed_strokes = []
        self.stroke_layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.background = np.zeros((height, width, 3), dtype=np.uint8)
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        self.camera_width = 320
        self.camera_height = 240

        self.prev_point = None
        self.points = []

        self.draw_grid()


    def draw_grid(self):

        self.background[:] = (55, 60, 70)

        grid_size = 15

        for x in range(0, self.width, grid_size):
            cv2.line(self.background, (x,0), (x,self.height), (85,90,100), 1)

        for y in range(0, self.height, grid_size):
            cv2.line(self.background, (0,y), (self.width,y), (85,90,100), 1)


    def draw_stroke(self, point):

        if self.prev_point is None:
            self.prev_point = point
            self.current_stroke.append(point)
            return

        dx = point[0] - self.prev_point[0]
        dy = point[1] - self.prev_point[1]

        dist = (dx*dx + dy*dy) ** 0.5

        if dist < 5:
            return

        smooth_x = int((point[0] + self.prev_point[0]) / 2)
        smooth_y = int((point[1] + self.prev_point[1]) / 2)

        smooth_point = (smooth_x, smooth_y)

        cv2.line(self.stroke_layer, self.prev_point, smooth_point, (255,255,255), 3)

        self.prev_point = smooth_point

        self.current_stroke.append(smooth_point)


        def reset_stroke(self):

            self.prev_point = None


    def render(self, frame):

        self.canvas[:] = self.background

        # add strokes
        self.canvas = cv2.add(self.canvas, self.stroke_layer)

        # draw camera preview
        cam = cv2.resize(frame, (self.camera_width, self.camera_height))
        self.canvas[0:self.camera_height, 0:self.camera_width] = cam

        return self.canvas
    
    def finish_stroke(self):

        if len(self.current_stroke) > 5:
            self.completed_strokes.append(self.current_stroke)

        self.current_stroke = []
        self.prev_point = None