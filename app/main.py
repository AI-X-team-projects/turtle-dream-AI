from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import json
from .core.pose_detector import PoseDetector

app = FastAPI(title="Posture Detection API")

# 정적 파일 서빙 설정
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 구체적인 origin을 지정해야 합니다
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_test_image():
    """테스트용 이미지 생성"""
    # 640x480 크기의 회색 이미지 생성
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    # 얼굴과 어깨를 표현하는 간단한 도형 그리기
    cv2.circle(img, (320, 150), 30, (200, 200, 200), -1)  # 머리
    cv2.circle(img, (300, 120), 5, (0, 0, 0), -1)  # 왼쪽 눈
    cv2.circle(img, (340, 120), 5, (0, 0, 0), -1)  # 오른쪽 눈
    cv2.circle(img, (320, 140), 5, (0, 0, 0), -1)  # 코
    cv2.line(img, (270, 250), (370, 250), (200, 200, 200), 20)  # 어깨
    return img

# 포스처 감지기 인스턴스 생성
detector = PoseDetector()

@app.get("/test-image")
async def get_test_image():
    """테스트 이미지 생성 및 반환"""
    img = create_test_image()
    _, buffer = cv2.imencode('.jpg', img)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return JSONResponse({
        "image": f"data:image/jpeg;base64,{base64_image}"
    })

@app.get("/")
async def root():
    """API 상태 확인"""
    return {"status": "running", "message": "포스처 감지 API가 실행 중입니다."}

@app.websocket("/ws/pose-detection")
async def pose_detection_websocket(websocket: WebSocket):
    """실시간 포스처 감지를 위한 웹소켓 엔드포인트"""
    print("새로운 웹소켓 연결 시도")
    await websocket.accept()
    print("웹소켓 연결 수락됨")
    
    try:
        while True:
            # 클라이언트로부터 이미지 데이터 수신
            # print("클라이언트로부터 데이터 대기 중...")
            data = await websocket.receive_text()
            # print("데이터 수신됨, 길이:", len(data))
            
            try:
                # base64 이미지 데이터 디코딩
                if not data.startswith('data:image'):
                    raise ValueError("잘못된 이미지 데이터 형식")
                    
                encoded_data = data.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    raise ValueError("이미지 디코딩 실패")
                    
                # print("이미지 디코딩 완료, 크기:", frame.shape)
                
                # 테스트를 위해 먼저 간단한 응답 보내기
                if frame.shape[0] == 1 or frame.shape[1] == 1:
                    # 테스트 이미지로 대체
                    frame = create_test_image()
                    print("테스트 이미지로 대체됨")
                
                # 포스처 분석 수행
                processed_frame, result = detector.analyze_frame(frame)
                print("포스처 분석 완료:", result["message"])
                
                # 처리된 이미지를 base64로 인코딩
                _, buffer = cv2.imencode('.jpg', processed_frame)
                processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 결과 전송
                await websocket.send_json({
                    "image": f"data:image/jpeg;base64,{processed_image_base64}",
                    "result": result
                })
                # print("결과 전송 완료")
                
            except Exception as e:
                print(f"처리 중 오류 발생: {str(e)}")
                await websocket.send_json({
                    "error": f"이미지 처리 중 오류 발생: {str(e)}"
                })
                
    except Exception as e:
        print(f"웹소켓 연결 오류: {str(e)}")
    finally:
        print("웹소켓 연결 종료")
        await websocket.close() 