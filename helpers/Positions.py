import numpy as np
import cv2
class Positions:
    def __init__(self):
        self.balls = []
        self.pockets = []

    def findCorners(self):
        # sort order of pockets in order going: top left, top right, bottom left, bottom right

        def sortAll(box):
            return box[1]
        def sort_corners(box):
            return box[0]
        self.pockets.sort(key=sortAll)
        # excludes middle pockets
        top_2 = self.pockets[0:2]

        bottom_2 = self.pockets[4:]
        top_2.sort(key=sort_corners)
        bottom_2.sort(key=sort_corners)

        return {"top_left" : top_2[0], "top_right" : top_2[1], "bottom_left" : bottom_2[0], "bottom_right" : bottom_2[1]}


    def initialTransform(self, pocket_coords):
        ''''get positions of pockets on inital image and applies transform to find their positions on birds eye image'''
        actual_pockets = self.findCorners()
        src = np.array((actual_pockets["bottom_left"], actual_pockets["bottom_right"], actual_pockets["top_left"],
                        actual_pockets["top_right"]), dtype=np.float32)
        dest = np.array(pocket_coords, dtype=np.float32)
        mtx = cv2.getPerspectiveTransform(src, dest)
        return mtx

    def separateBallsAndColors(self):
        balls = []
        colors = []
        for x in self.balls:
            balls.append(x.position)
            if x.color == "ball":
                colors.append("red")
            else:
                colors.append(x.color)
        return balls, colors

class Ball:
    def __init__(self, color, position):
        self.color = color
        self.position = position