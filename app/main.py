import os
import json
import base64
import cv2
import requests
import time
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .core.pose_detector import PoseDetector
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

app = FastAPI()
detector = PoseDetector()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PostureRequest(BaseModel):
    userId: str
    image: str

UPLOAD_DIR = "received_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze-posture")
async def analyze_posture(request: PostureRequest):
    """AI 서버에서 이미지 분석 후 백엔드에 저장"""
    try:
        if not request.image or not request.userId:
            raise HTTPException(status_code=400, detail="이미지 데이터 또는 사용자 ID가 없습니다.")

        print(f"받은 Base64 데이터 (앞 100자): {request.image[:100]}")

        # 이미지 Base64 디코딩
        received_base64 = request.image
        encoded_data = received_base64.split(",")[1] if "," in received_base64 else received_base64
        image_bytes = base64.b64decode(encoded_data)

        # 디버깅용
        image_path = f"{UPLOAD_DIR}/received_{request.userId}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        with open(image_path, "wb") as img_file:
            img_file.write(image_bytes)
        print(f"Base64 디코딩 성공, 저장 경로: {image_path}, 데이터 크기: {len(image_bytes)} bytes")

        # OpenCV를 사용하여 이미지 로드
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="이미지 디코딩 실패")
            
        # 이미지 크기 및 형태 로깅
        print(f"이미지 크기: {frame.shape}, 타입: {frame.dtype}")
        
        # 이미지에 타임스탬프 추가 (디버깅용)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # MediaPipe를 사용하여 자세 분석 실행
        processed_frame, analysis_result = detector.analyze_frame(frame)

        # 처리된 이미지 저장 (디버깅용)
        processed_image_path = f"{UPLOAD_DIR}/processed_{request.userId}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(processed_image_path, processed_frame)
        print(f"처리된 이미지 저장 경로: {processed_image_path}")

        # 처리된 이미지를 Base64로 인코딩하여 응답에 포함
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        processed_image_data_url = f"data:image/jpeg;base64,{processed_image_base64}"

        # 분석 결과 로깅
        analysis_result_copy = analysis_result.copy()
        if "landmarks" in analysis_result_copy:
            analysis_result_copy["landmarks"] = f"[{len(analysis_result_copy['landmarks'])} landmarks - 출력 생략]"
        print(f"AI 분석 결과: {json.dumps(analysis_result_copy, indent=2, ensure_ascii=False)}")

        # 백엔드에 데이터 저장 요청
        response = save_posture_data(request.userId, analysis_result)
        print(f"백엔드 저장 요청 완료, 응답 코드: {response.status_code}, 응답 메시지: {response.text}")

        # 응답에 처리된 이미지 포함
        return {
            "message": "이미지 분석 성공", 
            "analysis": analysis_result,
            "processed_image": processed_image_data_url
        }

    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"분석 실패: {e}")

def save_posture_data(user_id, result):
    """분석한 데이터를 백엔드에 저장하는 함수"""
    recorded_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload = {
        "userId": user_id,
        "isGoodPosture": result.get("is_good_posture", False),
        "postureStatus": result.get("posture_status", "UNKNOWN"),
        "feedback": result.get("feedback", "No feedback available"),
        "recordedAt": recorded_time,
        "badPostureDuration": result.get("bad_posture_duration", 0),
        "totalSessionDuration": result.get("total_session_duration", 0)
    }

    print(f"백엔드로 저장 요청: {BACKEND_URL}/api/posture/save")
    print(f"요청 데이터: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    print(f"저장 요청 - recordedAt: {recorded_time}")

    try:
        response = requests.post(f"{BACKEND_URL}/api/posture/save", json=payload)
        print(f"백엔드 저장 응답: {response.status_code}, {response.text}")
        return response
    except Exception as e:
        print(f"백엔드 저장 요청 실패: {e}")
        return None
