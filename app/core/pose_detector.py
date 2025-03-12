import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
import os
import platform
import time
from typing import Tuple, Dict, Any
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class PoseDetector:
    def __init__(self, model_path: str = 'pose_landmarker_heavy.task'):
        """포스처 감지기 초기화"""
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(self.options)
        
        # 알림 관련 변수 초기화
        self.bad_posture_start = None
        self.notification_cooldown = 0
        
    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임을 분석하고 결과를 반환합니다."""
        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 포즈 감지 수행
        detection_result = self.landmarker.detect(image_frame)
        
        # RGB를 BGR로 다시 변환
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        result = {
            "is_good_posture": True,
            "message": "바른 자세입니다!",
            "debug_info": "",
            "debug_info2": "",
            "landmarks": [],
            "notification": False
        }
        
        if detection_result.pose_landmarks:
            # 첫 번째 감지된 포즈에 대해 분석
            pose_landmarks = detection_result.pose_landmarks[0]
            
            # 랜드마크 그리기 및 좌표 저장
            h, w = frame.shape[:2]
            landmarks = []
            for landmark in pose_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                landmarks.append({"x": cx, "y": cy})
            
            # 자세 분석
            result = self._analyze_posture(pose_landmarks, frame.shape)
            result["landmarks"] = landmarks
            
            # 알림 처리
            if not result["is_good_posture"]:
                if self.bad_posture_start is None:
                    self.bad_posture_start = time.time()
                elif time.time() - self.bad_posture_start > 5 and self.notification_cooldown <= 0:
                    result["notification"] = True
                    self.notification_cooldown = 10  # 10초 동안 추가 알림 방지
            else:
                self.bad_posture_start = None
            
            if self.notification_cooldown > 0:
                self.notification_cooldown -= 1
            
            # 피드백 텍스트 추가
            color = (0, 255, 0) if result["is_good_posture"] else (0, 0, 255)
            frame = self._put_korean_text(frame, result["message"], (10, 30), 32, color)
            frame = self._put_korean_text(frame, result["debug_info"], (10, frame.shape[0] - 50), 20, (255, 255, 255))
            frame = self._put_korean_text(frame, result["debug_info2"], (10, frame.shape[0] - 25), 20, (255, 255, 255))
        
        return frame, result

    def _analyze_posture(self, pose_landmarks, image_shape) -> Dict[str, Any]:
        """자세를 분석하고 피드백을 제공합니다."""
        h, w = image_shape[:2]
        
        # 주요 랜드마크 포인트 추출
        nose = (int(pose_landmarks[0].x * w), int(pose_landmarks[0].y * h))
        left_eye = (int(pose_landmarks[2].x * w), int(pose_landmarks[2].y * h))
        right_eye = (int(pose_landmarks[5].x * w), int(pose_landmarks[5].y * h))
        left_shoulder = (int(pose_landmarks[11].x * w), int(pose_landmarks[11].y * h))
        right_shoulder = (int(pose_landmarks[12].x * w), int(pose_landmarks[12].y * h))
        
        # 눈과 어깨의 중심점 계산
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # 1. 머리 좌우 기울기 계산
        head_tilt = abs(left_eye[1] - right_eye[1])
        
        # 2. 거북목 판단을 위한 수직 비율 계산
        vertical_distance = abs(shoulder_center_y - eye_center_y)
        nose_to_eye_vertical = abs(nose[1] - eye_center_y)
        turtle_neck_ratio = nose_to_eye_vertical / vertical_distance if vertical_distance > 0 else 0
        
        # 임계값 설정
        tilt_threshold = h * 0.02
        turtle_neck_ratio_threshold = 0.15
        
        # 자세 분석
        message = ""
        is_good = True
        
        if head_tilt > tilt_threshold:
            message += "머리가 기울어져 있습니다. "
            is_good = False
        
        if turtle_neck_ratio > turtle_neck_ratio_threshold:
            message += "턱을 들어 목을 펴주세요. "
            is_good = False
            
        if is_good:
            message = "바른 자세입니다!"
            
        return {
            "is_good_posture": is_good,
            "message": message,
            "debug_info": f"좌우기울기: {head_tilt:.1f}/{tilt_threshold:.1f} | 거북목비율: {turtle_neck_ratio:.3f}/{turtle_neck_ratio_threshold:.3f}",
            "debug_info2": f"눈-어깨거리: {vertical_distance:.1f} | 코-눈거리: {nose_to_eye_vertical:.1f}",
            "metrics": {
                "head_tilt": head_tilt,
                "tilt_threshold": tilt_threshold,
                "turtle_neck_ratio": turtle_neck_ratio,
                "turtle_neck_ratio_threshold": turtle_neck_ratio_threshold,
                "vertical_distance": vertical_distance,
                "nose_to_eye_vertical": nose_to_eye_vertical
            }
        }

    def _put_korean_text(self, image: np.ndarray, text: str, position: Tuple[int, int], 
                        font_size: int, color: Tuple[int, int, int]) -> np.ndarray:
        """한글 텍스트를 이미지에 추가합니다."""
        pil_image = Image.fromarray(image)
        
        # OS에 따른 폰트 경로
        system = platform.system()
        if system == 'Windows':
            font_path = os.getenv('WINDOWS_FONT_PATH', 'C:/Windows/Fonts/malgun.ttf')
        else:  # macOS
            font_path = os.getenv('MACOS_FONT_PATH', '/System/Library/Fonts/AppleSDGothicNeo.ttc')
            if not os.path.exists(font_path):
                font_path = os.getenv('MACOS_FONT_PATH_BACKUP', '/System/Library/Fonts/Supplemental/AppleGothic.ttf')
        
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(pil_image)
        color = (color[2], color[1], color[0])  # BGR -> RGB
        draw.text(position, text, font=font, fill=color)
        
        return np.array(pil_image)

    def _get_font_path(self) -> str:
        """OS에 따른 폰트 경로를 반환합니다."""
        system = platform.system()
        if system == 'Windows':
            return os.getenv('WINDOWS_FONT_PATH')
        elif system == 'Darwin':  # macOS
            main_font = os.getenv('MACOS_FONT_PATH')
            if os.path.exists(main_font):
                return main_font
            return os.getenv('MACOS_FONT_PATH_BACKUP')
        return None 