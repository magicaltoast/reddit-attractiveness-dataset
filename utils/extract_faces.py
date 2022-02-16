from insightface.app import FaceAnalysis
from insightface.app.common import Face
from utils.align import FaceAligner
import sys
from scipy.spatial.distance import cosine
from statistics import mean


class FaceExtractor:
    def __init__(self, app, aligner, max_distance) -> None:
        self.app = app
        self.aligner = aligner
        self.max_distance = max_distance

    @classmethod
    def default(
        cls, app=None, aligner=None, max_distance=sys.maxsize, additional_modules=[]
    ):
        if not app:
            app = FaceAnalysis(
                name="buffalo_m",
                allowed_modules=["recognition", "detection", *additional_modules],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))

        if not aligner:
            aligner = FaceAligner()

        return cls(app, aligner, max_distance)

    def __call__(self, images, additional_modules=[]):
        single_detections = []
        multiple_detections = []
        for j, (image, _) in enumerate(images):
            bboxes, kpss = self.app.det_model.detect(image)
            faces = []

            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, 0:4]
                det_score = bboxes[i, 4]
                kps = None
                if kpss is not None:
                    kps = kpss[i]

                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                faces.append(face)

            if len(faces) == 1:
                single_detections.append((faces[0], j))
                for module in additional_modules:
                    self.app.models[module].get(image, faces[0])

            elif len(faces) > 1:
                multiple_detections.append((faces, j))

        if not single_detections:
            return [], [], []

        if multiple_detections:
            good_faces = []
            for good, i in single_detections:
                self.app.models["recognition"].get(images[i][0], good)
                good_faces.append((good, i))

            for m_group, i in multiple_detections:
                min_index = -1
                min_distance = sys.maxsize

                for j, bad in enumerate(m_group):
                    candidate = self.app.models["recognition"].get(images[i][0], bad)
                    distances = []

                    for good, _ in single_detections:
                        good_emb = good["embedding"]
                        distances.append(cosine(good_emb, candidate))

                    mean_distance = mean(distances)

                    if mean_distance < min_distance:
                        min_distance = mean_distance
                        min_index = j

                if min_distance < self.max_distance:
                    best_face = m_group[min_index]
                    good_faces.append((best_face, i))

                    for module in additional_modules:
                        self.app.models[module].get(images[i][0], best_face)

            for face in good_faces:
                face[0].pop("embedding")
        else:
            good_faces = single_detections

        detections = []
        aligned = []
        urls = []

        for face in good_faces:
            detection = face[0]
            detections.append(detection)
            urls.append(images[face[1]][1])
            img = images[face[1]][0]
            aligned.append(self.aligner.from_insight_face(img, [detection])[0])

        return detections, aligned, urls
