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
        
        # 나쁜 자세 감지 관련 변수
        self.bad_posture_start = None
        self.notification_cooldown = 0
        
        # 전체 세션 시간 기록 (앱 실행 후 초기화)
        self.session_start_time = None  

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """프레임을 분석하고 결과를 반환합니다."""
        
        if self.session_start_time is None:
            self.session_start_time = time.time()  # 전체 세션 시작 시간 설정

        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 포즈 감지 수행
        detection_result = self.landmarker.detect(image_frame)
        
        # RGB를 BGR로 다시 변환
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # 기본값 설정
        result = {
            "is_good_posture": True,
            "posture_status": "GOOD",
            "feedback": "바른 자세입니다!",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "bad_posture_duration": 0,
            "total_session_duration": int(time.time() - self.session_start_time),
            "notification": False
        }

        if detection_result.pose_landmarks:
            pose_landmarks = detection_result.pose_landmarks[0]
            
            # 자세 분석 수행
            result = self._analyze_posture(pose_landmarks, frame.shape)

            # 나쁜 자세 유지 시간 계산
            if not result["is_good_posture"]:
                if self.bad_posture_start is None:
                    self.bad_posture_start = time.time()
                result["bad_posture_duration"] = int(time.time() - self.bad_posture_start)
            else:
                self.bad_posture_start = None  # 올바른 자세로 돌아오면 초기화

            # 알림 처리 (5초 이상 BAD 자세 지속 시 알림)
            if not result["is_good_posture"] and result["bad_posture_duration"] >= 5 and self.notification_cooldown <= 0:
                result["notification"] = True
                self.notification_cooldown = 10  # 10초 동안 추가 알림 방지

            if self.notification_cooldown > 0:
                self.notification_cooldown -= 1
            
            # 랜드마크 저장
            h, w = frame.shape[:2]
            landmarks = []
            for landmark in pose_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                landmarks.append({"x": cx, "y": cy})
            result["landmarks"] = landmarks

            # 피드백 텍스트 추가
            color = (0, 255, 0) if result["is_good_posture"] else (0, 0, 255)
            frame = self._put_korean_text(frame, result["feedback"], (10, 30), 32, color)

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

        # 자세 분석
        head_tilt = abs(left_eye[1] - right_eye[1])
        vertical_distance = abs(left_shoulder[1] - right_shoulder[1])
        turtle_neck_ratio = abs(nose[1] - (left_eye[1] + right_eye[1]) / 2) / vertical_distance

        tilt_threshold = h * 0.02
        turtle_neck_ratio_threshold = 0.15

        message = "바른 자세입니다!"
        is_good = True

        if head_tilt > tilt_threshold:
            message = "머리가 기울어져 있습니다."
            is_good = False

        if turtle_neck_ratio > turtle_neck_ratio_threshold:
            message = "턱을 들어 목을 펴주세요."
            is_good = False

        return {
            "is_good_posture": is_good,
            "posture_status": "GOOD" if is_good else "BAD",
            "feedback": message,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "bad_posture_duration": 0,  # analyze_frame에서 업데이트됨
            "total_session_duration": 0  # analyze_frame에서 업데이트됨
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