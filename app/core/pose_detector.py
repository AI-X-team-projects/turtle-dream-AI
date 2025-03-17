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
        
        # ----------------------------------------------------------------------------------------------------
        # 포즈 상태 지속 시간 기록(dev에 반영 X)
        # 현재 자세 상태(posture_status)를 이전 상태(prev_posture_status)와 비교.
        # 상태가 변경되었으면 즉시 저장.
        # 상태가 1분 이상 유지되면 30초마다 저장.
        # self.last_saved_time = 0  # 마지막 저장된 시간
        # self.posture_status_duration = 0  # 현재 자세 상태 지속 시간
        # self.prev_posture_status = None  # 이전 자세 상태
        # ----------------------------------------------------------------------------------------------------

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
            
            # ----------------------------------------------------------------------------------------------------
            # 포즈 상태 지속 시간 기록(dev에 반영 X)
            # current_time = time.time()
            # current_status = result["posture_status"]
            
            # # 1상태가 바뀌면 즉시 저장
            # if current_status != self.prev_posture_status:
            #     self.posture_status_duration = 0  # 지속 시간 초기화
            #     self.last_saved_time = current_time  # 즉시 저장
            #     save_posture_data("USER_ID", result)  # 백엔드 저장 요청
            #     print(f"상태 변경 감지: {current_status} → 즉시 저장")

            # # 상태가 1분 이상 유지되면 30초마다 저장
            # else:
            #     self.posture_status_duration += 1  # 1프레임마다 증가 (초 단위로 변환됨)
                
            #     if self.posture_status_duration >= 60 and (current_time - self.last_saved_time) >= 30:
            #         save_posture_data("USER_ID", result)
            #         self.last_saved_time = current_time  # 마지막 저장 시간 업데이트
            #         print(f"{current_status} 유지 1분 이상 → 30초마다 저장")

            # # 이전 상태 업데이트
            # self.prev_posture_status = current_status
            # ----------------------------------------------------------------------------------------------------

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
        eye_center = (eye_center_x, eye_center_y)
        
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_center = (shoulder_center_x, shoulder_center_y)

        # 자세 분석 - 핵심 지표 계산
        head_tilt = abs(left_eye[1] - right_eye[1])  # 머리 좌우 기울기
        shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])  # 어깨 기울기
        
        # 머리와 어깨의 기울기 비율 계산 (머리 기울기가 어깨 기울기보다 크면 머리만 기울어진 것)
        head_shoulder_tilt_ratio = 1.0
        if shoulder_tilt > 0:
            head_shoulder_tilt_ratio = head_tilt / shoulder_tilt
        
        # 턱 좌표 (인덱스 17이 없을 수 있으므로 예외 처리)
        try:
            chin = (int(pose_landmarks[17].x * w), int(pose_landmarks[17].y * h))  # 턱 좌표
            face_length = abs(nose[1] - chin[1])  # 얼굴 길이
        except IndexError:
            print("경고: 턱 랜드마크(17)를 찾을 수 없습니다. face_length를 0으로 설정합니다.")
            chin = nose  # 기본값으로 코 위치 사용
            face_length = 0
        
        # 거북목 비율 계산 (코-눈 거리 / 눈-어깨 거리)
        eye_to_shoulder_distance = abs(eye_center_y - shoulder_center_y)
        nose_to_eye_vertical = abs(nose[1] - eye_center_y)
        
        if eye_to_shoulder_distance > 0:
            turtle_neck_ratio = nose_to_eye_vertical / eye_to_shoulder_distance
        else:
            turtle_neck_ratio = 0
            print("경고: eye_to_shoulder_distance가 0입니다. turtle_neck_ratio를 0으로 설정합니다.")
        
        # 머리 높이 비율 (코 높이 / 어깨 높이)
        if shoulder_center_y > 0:
            head_height_ratio = nose[1] / shoulder_center_y
        else:
            head_height_ratio = 1.0
            print("경고: shoulder_center_y가 0입니다. head_height_ratio를 1.0으로 설정합니다.")
        
        # 임계값 설정 - 더 높게 조정
        head_tilt_threshold = h * 0.08  # 머리 좌우 기울기 임계값 (화면 높이의 8%)
        turtle_neck_ratio_threshold = 10.0  # 거북목 비율 임계값
        shoulder_tilt_threshold = h * 0.08  # 어깨 기울기 임계값 (화면 높이의 8%)
        height_threshold = 1.05  # 머리 높이 비율 임계값 (5% 이상 낮으면 거북목)
        face_length_threshold = 530  # 얼굴 길이 임계값 (530px 이상이면 거북목)
        head_shoulder_ratio_threshold = 3.0  # 머리 기울기가 어깨 기울기의 3배 이상이면 머리만 기울어진 것
        
        # 디버깅 로그
        print(f"자세 분석 - 머리 좌우 기울기: {head_tilt:.1f}/{head_tilt_threshold:.1f}")
        print(f"자세 분석 - 거북목 비율: {turtle_neck_ratio:.3f}/{turtle_neck_ratio_threshold:.3f}")
        print(f"자세 분석 - 눈-어깨 거리: {eye_to_shoulder_distance:.1f}")
        print(f"자세 분석 - 코-눈 거리: {nose_to_eye_vertical:.1f}")
        print(f"자세 분석 - 얼굴 길이: {face_length:.1f}/{face_length_threshold:.1f}")
        print(f"자세 분석 - 어깨 기울기: {shoulder_tilt:.1f}/{shoulder_tilt_threshold:.1f}")
        print(f"자세 분석 - 머리/어깨 기울기 비율: {head_shoulder_tilt_ratio:.2f}/{head_shoulder_ratio_threshold:.1f}")
        print(f"자세 분석 - 머리 높이 비율: {head_height_ratio:.3f}/{height_threshold:.3f}")

        # 자세 분석 결과
        message = "바른 자세입니다!"
        is_good = True
        reason = ""

        # 자세 분석 조건 검사 - 로직 개선
        is_head_tilted = head_tilt > head_tilt_threshold
        is_shoulder_tilted = shoulder_tilt > shoulder_tilt_threshold
        
        # 머리와 어깨 기울기 판단 로직 개선
        if is_head_tilted and is_shoulder_tilted:
            # 둘 다 기울어진 경우, 어느 쪽이 더 심한지 판단
            if head_shoulder_tilt_ratio > head_shoulder_ratio_threshold:
                message = "머리가 많이 기울어져 있습니다. 고개를 바로 하세요."
                is_good = False
                reason += "머리 기울기 "
            else:
                message = "전체적인 자세가 기울어져 있습니다. 바른 자세를 유지하세요."
                is_good = False
                reason += "전체 자세 기울기 "
        elif is_head_tilted:
            # 머리만 기울어진 경우
            message = "머리가 기울어져 있습니다. 고개를 바로 하세요."
            is_good = False
            reason += "머리 기울기 "
        elif is_shoulder_tilted:
            # 어깨만 기울어진 경우
            message = "어깨가 기울어져 있습니다. 균형을 맞추세요."
            is_good = False
            reason += "어깨 기울기 "

        if turtle_neck_ratio > turtle_neck_ratio_threshold:
            message = "턱을 들어 목을 펴주세요."
            is_good = False
            reason += "거북목 비율 "
            
        if face_length > face_length_threshold:
            message = "얼굴이 너무 가깝습니다. 거북목 자세입니다."
            is_good = False
            reason += "얼굴 길이 거북목 "
            
        if head_height_ratio > height_threshold:
            message = "머리가 너무 낮습니다. 자세를 교정하세요."
            is_good = False
            reason += "머리 높이 "
        
        print(f"자세 분석 결과: {'나쁨' if not is_good else '좋음'}, 이유: {reason if reason else '없음'}")
    
        # 결과 반환
        return {
            "is_good_posture": is_good,
            "posture_status": "GOOD" if is_good else "BAD",
            "feedback": message,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "bad_posture_duration": 0,  # analyze_frame에서 업데이트됨
            "total_session_duration": 0,  # analyze_frame에서 업데이트됨
            "posture_metrics": {
                "head_tilt": head_tilt,
                "turtle_neck_ratio": turtle_neck_ratio,
                "face_length": face_length,
                "shoulder_tilt": shoulder_tilt,
                "head_height_ratio": head_height_ratio,
                "eye_to_shoulder_distance": eye_to_shoulder_distance,
                "nose_to_eye_vertical": nose_to_eye_vertical
            },
            "thresholds": {
                "head_tilt_threshold": head_tilt_threshold,
                "turtle_neck_ratio_threshold": turtle_neck_ratio_threshold,
                "face_length_threshold": face_length_threshold,
                "shoulder_tilt_threshold": shoulder_tilt_threshold,
                "height_threshold": height_threshold
            },
            "reason": reason
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