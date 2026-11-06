# coding: utf-8

"""
Face detection and initial landmarks with MediaPipe Face Mesh (Apache-2.0).

Replaces the InsightFace detector (buffalo_l: det_10g + 2d106det), whose models are
licensed for non-commercial research only. LivePortrait only needs an initial set of
landmarks here: it crops the face with them and then runs its own 203-point landmark
model (pretrained_weights/liveportrait/landmark.onnx, MIT).

Each face carries `bbox`, `det_score` and `landmark_2d`: 68 points in the dlib/iBUG
order, which src/utils/crop.py already understands (parse_pt2_from_pt68).
"""

import numpy as np
import cv2
import mediapipe as mp

from .rprint import rlog as log
from .timer import Timer

# MediaPipe Face Mesh index for each of the 68 iBUG points: jaw (0-16), brows (17-26),
# nose (27-35), eyes (36-47), mouth (48-67).
MP_TO_68 = np.array([
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    71, 63, 105, 66, 107, 336, 296, 334, 293, 301,
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87,
], dtype=np.int32)

DEFAULT_MAX_FACES = 10


class Face(dict):
    """Same shape as InsightFace's Face: a dict whose keys are also attributes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def sort_by_direction(faces, direction: str = 'large-small', face_center=None):
    if len(faces) <= 0:
        return faces

    if direction == 'left-right':
        return sorted(faces, key=lambda face: face['bbox'][0])
    if direction == 'right-left':
        return sorted(faces, key=lambda face: face['bbox'][0], reverse=True)
    if direction == 'top-bottom':
        return sorted(faces, key=lambda face: face['bbox'][1])
    if direction == 'bottom-top':
        return sorted(faces, key=lambda face: face['bbox'][1], reverse=True)
    if direction == 'small-large':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
    if direction == 'large-small':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]), reverse=True)
    if direction == 'distance-from-retarget-face':
        return sorted(faces, key=lambda face: (((face['bbox'][2]+face['bbox'][0])/2-face_center[0])**2+((face['bbox'][3]+face['bbox'][1])/2-face_center[1])**2)**0.5)
    return faces


class FaceAnalysisMediaPipe:
    def __init__(self, det_thresh: float = 0.5, max_faces: int = DEFAULT_MAX_FACES):
        self.timer = Timer()
        self.det_thresh = det_thresh
        self.max_faces = max_faces
        self._meshes = {}

    def _mesh(self, max_faces: int):
        # One Face Mesh per face limit, built on first use. Static mode: every call detects
        # from scratch, like the InsightFace detector did (LivePortrait tracks frames itself).
        if max_faces not in self._meshes:
            self._meshes[max_faces] = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=max_faces,
                refine_landmarks=False,
                min_detection_confidence=self.det_thresh,
            )
        return self._meshes[max_faces]

    def get(self, img_bgr, **kwargs):
        max_num = kwargs.get('max_face_num', 0)  # 0 means no limit (capped at max_faces)
        direction = kwargs.get('direction', 'large-small')
        max_faces = max_num if max_num and max_num > 0 else self.max_faces

        h, w = img_bgr.shape[:2]
        img_rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        result = self._mesh(max_faces).process(img_rgb)
        if not result.multi_face_landmarks:
            return []

        faces = []
        for face_landmarks in result.multi_face_landmarks:
            pts = np.array([(p.x * w, p.y * h) for p in face_landmarks.landmark], dtype=np.float32)
            lmk68 = pts[MP_TO_68]
            bbox = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=np.float32)
            # Face Mesh gives no per-face score; it only returns faces above det_thresh.
            faces.append(Face(bbox=bbox, det_score=1.0, landmark_2d=lmk68))

        return sort_by_direction(faces, direction)

    def warmup(self):
        self.timer.tic()
        self.get(np.zeros((512, 512, 3), dtype=np.uint8))
        log(f'FaceAnalysisMediaPipe warmup time: {self.timer.toc():.3f}s')
