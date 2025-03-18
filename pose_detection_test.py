# 기존 import문들은 그대로 유지
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

# WebSocket 관련 import 추가
import asyncio
import websockets
import json
import base64

# 기존의 모든 함수들(get_font_path, signal_handler, put_korean_text, analyze_posture)은 그대로 유지

def process_frame(landmarker, image):
    """단일 프레임을 처리하는 함수"""
    # BGR을 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # 포즈 감지 수행
    detection_result = landmarker.detect(image_frame)
    
    # RGB를 BGR로 다시 변환
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    result = {
        'success': False,
        'message': '포즈가 감지되지 않았습니다.'
    }

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

        result = {
            'success': True,
            'isGoodPosture': is_good_posture,
            'feedback': feedback,
            'debugInfo': debug_info,
            'debugInfo2': debug_info2
        }

    return image, result

async def websocket_handler(websocket, path):
    """WebSocket 연결을 처리하는 함수"""
    try:
        # MediaPipe Pose 모델 초기화
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                try:
                    # 웹소켓으로부터 이미지 데이터 수신
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # base64 이미지 데이터를 numpy 배열로 변환
                    img_data = base64.b64decode(data['image'])
                    nparr = np.frombuffer(img_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # 이미지 처리
                    processed_image, result = process_frame(landmarker, image)
                    
                    # 결과를 클라이언트로 전송
                    await websocket.send(json.dumps(result))

                except websockets.exceptions.ConnectionClosed:
                    print("클라이언트 연결이 종료되었습니다.")
                    break
                except Exception as e:
                    print(f"처리 중 오류 발생: {e}")
                    await websocket.send(json.dumps({
                        'success': False,
                        'message': str(e)
                    }))
    except Exception as e:
        print(f"서버 오류: {e}")

def main():
    """메인 함수 - 웹캠 모드와 웹소켓 모드 선택 가능"""
    try:
        mode = input("실행 모드를 선택하세요 (1: 웹캠, 2: 웹소켓): ")
        
        if mode == "1":
            # 기존의 웹캠 모드 코드
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

                    # 프레임 처리
                    processed_image, result = process_frame(landmarker, image)
                    
                    # 나쁜 자세 알림 처리
                    if result['success']:
                        if not result['isGoodPosture']:
                            if bad_posture_start is None:
                                bad_posture_start = time.time()
                            elif time.time() - bad_posture_start > 5 and notification_cooldown <= 0:
                                print("\a")  # 비프음 재생
                                notification_cooldown = 10
                        else:
                            bad_posture_start = None
                        
                        if notification_cooldown > 0:
                            notification_cooldown -= 1

                    # 결과 화면 표시
                    cv2.imshow('Pose Detection', processed_image)
                    
                    # 'q' 키를 누르면 종료
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

        elif mode == "2":
            # 웹소켓 서버 모드
            print("웹소켓 서버를 시작합니다...")
            start_server = websockets.serve(
                websocket_handler,
                "localhost",
                8001
            )
            asyncio.get_event_loop().run_until_complete(start_server)
            print("AI 분석 서버가 시작되었습니다. (ws://localhost:8001)")
            asyncio.get_event_loop().run_forever()

    except Exception as e:
        print(f"에러가 발생했습니다: {e}")
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    main()