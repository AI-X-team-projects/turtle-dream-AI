import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
from PIL import Image, ImageDraw, ImageFont
import os
import signal
import platform
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def get_font_path():
    """OS에 따른 폰트 경로를 반환합니다."""
    system = platform.system()
    if system == 'Windows':
        return os.getenv('WINDOWS_FONT_PATH')
    elif system == 'Darwin':  # macOS
        main_font = os.getenv('MACOS_FONT_PATH')
        if os.path.exists(main_font):
            return main_font
        return os.getenv('MACOS_FONT_PATH_BACKUP')
    else:
        return None

def signal_handler(sig, frame):
    """Ctrl+C로 프로그램 종료 시 처리"""
    print('\n프로그램을 종료합니다.')
    cv2.destroyAllWindows()
    os._exit(0)

# Ctrl+C 시그널 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)

def put_korean_text(image, text, position, font_size, color):
    """한글 텍스트를 이미지에 추가합니다."""
    # PIL 이미지로 변환
    pil_image = Image.fromarray(image)
    
    # OS에 따른 폰트 경로 가져오기
    font_path = get_font_path()
    
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # 폰트 파일이 없는 경우 기본 폰트 사용
            print("경고: 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            font = ImageFont.load_default()
    except Exception as e:
        print(f"폰트 로드 중 오류 발생: {e}")
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(pil_image)
    
    # RGB -> BGR 색상 변환
    color = (color[2], color[1], color[0])
    
    # 텍스트 그리기
    draw.text(position, text, font=font, fill=color)
    
    # numpy 배열로 다시 변환
    return np.array(pil_image)

def analyze_posture(pose_landmarks, image_shape):
    """자세를 분석하고 피드백을 제공합니다."""
    h, w = image_shape[:2]
    
    # 주요 랜드마크 포인트 추출
    nose = (int(pose_landmarks[0].x * w), int(pose_landmarks[0].y * h))  # 코
    left_eye = (int(pose_landmarks[2].x * w), int(pose_landmarks[2].y * h))  # 왼쪽 눈
    right_eye = (int(pose_landmarks[5].x * w), int(pose_landmarks[5].y * h))  # 오른쪽 눈
    left_shoulder = (int(pose_landmarks[11].x * w), int(pose_landmarks[11].y * h))  # 왼쪽 어깨
    right_shoulder = (int(pose_landmarks[12].x * w), int(pose_landmarks[12].y * h))  # 오른쪽 어깨
    
    # 눈과 어깨의 중심점 계산
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
    
    # 1. 머리 좌우 기울기 계산 (눈의 수평 기울기)
    head_tilt = abs(left_eye[1] - right_eye[1])
    
    # 2. 거북목 판단을 위한 수직 비율 계산
    # 눈-어깨 사이의 수직 거리
    vertical_distance = abs(shoulder_center_y - eye_center_y)
    
    # 코-눈 사이의 수직 거리 (양수면 코가 눈보다 아래에 있다는 의미)
    nose_to_eye_vertical = abs(nose[1] - eye_center_y)
    
    # 거북목 비율 = (코-눈 수직거리) / (눈-어깨 수직거리)
    # 이 값이 클수록 거북목이 심하다는 의미
    turtle_neck_ratio = nose_to_eye_vertical / vertical_distance if vertical_distance > 0 else 0
    
    # 거북목 판단을 위한 임계값
    tilt_threshold = h * 0.02  # 화면 높이의 2%
    turtle_neck_ratio_threshold = 0.15  # 거북목 비율 임계값 (15% 초과시 거북목으로 판단)
    
    # 자세 상태 분석
    message = ""
    is_good = True
    color = (0, 255, 0)  # 기본 녹색
    
    # 1. 머리 좌우 기울기 체크
    if head_tilt > tilt_threshold:
        message += "머리가 기울어져 있습니다. "
        is_good = False
    
    # 2. 거북목 체크 (코가 눈보다 얼마나 아래에 있는지)
    # 거북목일수록 코가 눈보다 더 아래로 내려가므로 비율이 커짐
    if turtle_neck_ratio > turtle_neck_ratio_threshold:
        message += "턱을 들어 목을 펴주세요. "
        is_good = False
    
    # 디버깅 정보 표시
    debug_info = f"좌우기울기: {head_tilt:.1f}/{tilt_threshold:.1f} | 거북목비율: {turtle_neck_ratio:.3f}/{turtle_neck_ratio_threshold:.3f}"
    
    # 추가 디버깅 정보
    debug_info2 = f"눈-어깨거리: {vertical_distance:.1f} | 코-눈거리: {nose_to_eye_vertical:.1f}"
    
    # 최종 판단
    if is_good:
        return True, "바른 자세입니다!", (0, 255, 0), debug_info, debug_info2
    else:
        return False, f"자세 교정 필요: {message}", (0, 0, 255), debug_info, debug_info2

def main():
    try:
        # MediaPipe Pose 모델 초기화
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # 웹캠 초기화
        cap = cv2.VideoCapture(0)
        
        # 자세 알림을 위한 변수들
        bad_posture_start = None
        notification_cooldown = 0
        
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("웹캠을 찾을 수 없습니다.")
                    break

                # BGR을 RGB로 변환
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

                # 포즈 감지 수행
                detection_result = landmarker.detect(image_frame)
                
                # RGB를 BGR로 다시 변환
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # 감지된 포즈가 있다면 분석 수행
                if detection_result.pose_landmarks:
                    # 첫 번째 감지된 포즈에 대해 분석
                    pose_landmarks = detection_result.pose_landmarks[0]
                    
                    # 랜드마크 그리기
                    for landmark in pose_landmarks:
                        h, w, _ = image.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                    
                    # 자세 분석
                    is_good_posture, feedback, color, debug_info, debug_info2 = analyze_posture(pose_landmarks, image.shape)
                    
                    # 한글 피드백 표시
                    image = put_korean_text(image, feedback, (10, 30), 32, color)
                    # 디버깅 정보 표시
                    image = put_korean_text(image, debug_info, (10, image.shape[0] - 50), 20, (255, 255, 255))
                    image = put_korean_text(image, debug_info2, (10, image.shape[0] - 25), 20, (255, 255, 255))
                    
                    # 나쁜 자세 지속 시간 체크 및 알림
                    if not is_good_posture:
                        if bad_posture_start is None:
                            bad_posture_start = time.time()
                        elif time.time() - bad_posture_start > 5 and notification_cooldown <= 0:  # 5초 이상 나쁜 자세 유지
                            print("\a")  # 비프음 재생
                            notification_cooldown = 10  # 10초 동안 추가 알림 방지
                    else:
                        bad_posture_start = None
                    
                    if notification_cooldown > 0:
                        notification_cooldown -= 1

                # 결과 화면 표시
                cv2.imshow('Pose Detection', image)
                
                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"에러가 발생했습니다: {e}")
    finally:
        # 자원 해제
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # 추가 대기 시간
        
if __name__ == "__main__":
    main()
    # 프로그램 종료 시 모든 창 닫기
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # 추가 대기 시간 