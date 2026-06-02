import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from config import (
    HAND_LANDMARKER_PATH              as _LANDMARKER_PATH,
    CAMERA_INDEX                      as _DEFAULT_CAMERA,
    DETECTOR_MIN_DETECTION_CONFIDENCE as _DEFAULT_DETECTION,
    DETECTOR_MIN_PRESENCE_CONFIDENCE  as _DEFAULT_PRESENCE,
    DETECTOR_MIN_TRACKING_CONFIDENCE  as _DEFAULT_TRACKING,
)

# BGR drawing colors
_COLOR_LANDMARK = (0, 217, 255)
_COLOR_CONNECTION = (255, 255, 255)
_COLOR_OK = (0, 255, 0)
_COLOR_NONE = (0, 0, 200)
_COLOR_HUD = (255, 255, 0)


class Detector:
    def __init__(
        self,
        camera_index=_DEFAULT_CAMERA,
        min_detection_confidence=_DEFAULT_DETECTION,
        min_presence_confidence=_DEFAULT_PRESENCE,
        min_tracking_confidence=_DEFAULT_TRACKING,
    ):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_LANDMARKER_PATH),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._connections = list(HandLandmarksConnections.HAND_CONNECTIONS)

        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open the camera (index {camera_index}).")

        self._start_time = time.time()

    def get_frame(self):
        """
        Capture a frame, detect landmarks and return them with the annotated frame.

        Returns:
            frame (np.ndarray | None): BGR image with landmarks drawn on it.
            landmarks_data (dict): {
                "num_hands": int,
                "landmarks_hand1": list[float] | None,  # 63 values (21 x,y,z)
                "landmarks_hand2": list[float] | None,
            }
        """
        ret, frame = self._cap.read()
        if not ret:
            return None, self._empty_result()

        frame = cv2.flip(frame, 1)

        timestamp_ms = int((time.time() - self._start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks_data = self._extract_landmarks(result)
        self._draw_landmarks(frame, result)
        self._draw_hud(frame, landmarks_data)

        return frame, landmarks_data

    def _extract_landmarks(self, result):
        data = self._empty_result()

        if not result.hand_landmarks:
            return data

        data["num_hands"] = len(result.hand_landmarks)

        for i, hand in enumerate(result.hand_landmarks):
            flat = []
            for lm in hand:
                flat.extend([lm.x, lm.y, lm.z])
            if i == 0:
                data["landmarks_hand1"] = flat
            elif i == 1:
                data["landmarks_hand2"] = flat

        return data

    def _draw_landmarks(self, frame, result):
        if not result.hand_landmarks:
            return

        h, w = frame.shape[:2]

        for hand in result.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            for conn in self._connections:
                cv2.line(frame, points[conn.start], points[conn.end], _COLOR_CONNECTION, 2, cv2.LINE_AA)

            for pt in points:
                cv2.circle(frame, pt, 5, _COLOR_LANDMARK, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_hud(self, frame, landmarks_data):
        num = landmarks_data["num_hands"]
        color = _COLOR_OK if num > 0 else _COLOR_NONE
        cv2.putText(frame, f"Hands: {num}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    def _empty_result(self):
        return {"num_hands": 0, "landmarks_hand1": None, "landmarks_hand2": None}

    def release(self):
        self._cap.release()
        self._landmarker.close()
