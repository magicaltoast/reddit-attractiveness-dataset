from math import cos, sin, degrees, radians
import cv2
import numpy as np
from functools import partial


class FaceAligner:
    def __init__(self, respect_old_bounding_box=True, square=True, padding=1.25):
        self.respect_old_bounding_box = respect_old_bounding_box
        self.square = square
        self.padding = padding

    def size_after_rotation(self, width, height, angle):
        angle_sin = abs(sin(angle))
        angle_cos = abs(cos(angle))
        return (
            width * angle_cos + height * angle_sin,
            width * angle_sin + height * angle_cos,
        )

    def align(self, img, bbox, left_eye, right_eye):
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        center = (x1 + x2) / 2, (y1 + y2) / 2

        angle = self.get_angle(left_eye, right_eye)

        if self.respect_old_bounding_box:
            width, height = self.size_after_rotation(width, height, angle)

        if self.square:
            width = height = max(width, height)

        if self.padding:
            width *= self.padding
            height *= self.padding

        rot_mat = cv2.getRotationMatrix2D(center, degrees(angle), 1)
        rot_mat[0, 2] += (width * 0.5) - center[0]
        rot_mat[1, 2] += (height * 0.5) - center[1]

        return cv2.warpAffine(img, rot_mat, (int(width), int(height)))

    def _to_int(self, array):
        return tuple(map(int, array))

    def visualize(
        self,
        img,
        bbox,
        left_eye,
        right_eye,
        color=(255, 255, 255),
    ):
        img = cv2.rectangle(
            img, self._to_int(bbox[:2]), self._to_int(bbox[2:]), color, 10
        )

        for landmark in [left_eye, right_eye]:
            img = cv2.circle(img, tuple(landmark), 10, color)

        img = self.align(img, bbox, left_eye, right_eye)

        return img

    def get_angle(self, left_eye, right_eye):
        d_y = right_eye[1] - left_eye[1]
        d_x = right_eye[0] - left_eye[0]
        return np.arctan2(d_y, d_x) - radians(180)

    def from_retina_face(self, img, result, viz=False):
        res = []
        aling_func = self.visualize if viz else self.align
        for face in result.values():
            landmarks = face["landmarks"]
            bbox = face["facial_area"]
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            res.append(aling_func(img, bbox, left_eye, right_eye))
        return res

    def from_mtcnn(self, img, result, viz=False):
        res = []
        aling_func = self.visualize if viz else self.align
        for face in result:
            x, y, w, h = face["box"]
            bbox = [x, y, x + w, y + h]
            landmarks = face["keypoints"]
            left_eye = landmarks["right_eye"]
            right_eye = landmarks["left_eye"]
            res.append(aling_func(img, bbox, left_eye, right_eye))
        return res

    @staticmethod
    def normalize_point(height, width, point):
        return (min(max(point[0], 0), width), min(max(point[1], 0), height))

    def from_insight_face(self, img, result, viz=False):
        res = []
        aling_func = self.visualize if viz else self.align
        normalize = partial(self.normalize_point, *img.shape[:2])
        for face in result:
            landmarks = face["kps"].astype("int")[:2].tolist()
            right_eye, left_eye = map(normalize, landmarks)
            x1, y1, x2, y2 = face["bbox"].tolist()
            bbox = [*normalize((x1, y1)), *normalize((x2, y2))]
            res.append(aling_func(img, bbox, left_eye, right_eye))
        return res
