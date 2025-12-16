from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from starlette.middleware.cors import CORSMiddleware
from llm_service import LLM_gemini
from pydantic import BaseModel

# 실시간 전송용
import json
import asyncio
import re
from toxic_detector import ToxicClauseDetector
from fastapi.responses import StreamingResponse # 스트리밍 응답용

app = FastAPI()

# 통신을 허용할 포트 선택
origins = [
    "http://127.0.0.1:5173","http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,    # 허용 명단만 통과
    allow_credentials = True,   # 쿠키(로그인 정보 등) 주고받기 허용
    allow_methods = ["*"],      # GET, POST 등 모든 방식 허용
    allow_headers=["*"],        # 어떤 헤더 정보도 허용
)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...) ,api_key: str = Form(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    pdf_bytes = await file.read()
    try:
        extractor = LLM_gemini(gemini_api_key=api_key,model="gemini-2.0-flash-lite")
        extracted_text = extractor.pdf_to_text(pdf_bytes)

        # 4. 결과 반환 (JSON)
        return {
            "status": "success",
            "filename": file.filename,
            "text": extracted_text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def parse_text_to_chunks(text):
    """텍스트를 '제N조' 기준으로 자르는 파서"""
    if not text:
        return []
    split_pattern = r'(?=\n\s*제\s*\d+\s*조)'
    chunks = re.split(split_pattern, text)
    # 공백 제거 및 유효한 조항만 필터링
    clean_chunks = [c.strip() for c in chunks if len(c.strip()) > 10]
    return clean_chunks

class AnalyzeRequest(BaseModel):
    api_key: str
    text: str
@app.post("/analyze")
async def analyze_contract(request: AnalyzeRequest):
    # 제너레이터 함수: 데이터를 조금씩 나누어 보냅니다.
    async def event_stream():
        try:
            yield json.dumps({"status": "progress","message": "법령, 판례 DB 불러오는 중..."}) + "\n"
            detector = ToxicClauseDetector(api_key=request.api_key)
            chunks  = parse_text_to_chunks(request.text)
            yield json.dumps({"status": "progress","message": f"총 {len(chunks)}개의 조항을 분석 중..."}) + "\n"
            raw_results = detector.detect(chunks, max_concurrent=5)

            processed_results = []
            toxic_indices = [] # 개선안 생성이 필요한 인덱스들

            for i, res in enumerate(raw_results):
                # detect 함수에서 나온 결과에 ID(조항 번호) 추가
                res['id'] = i + 1
                res['suggestion'] = "" # 초기화
                processed_results.append(res)
            
                if res['is_toxic']:
                    toxic_indices.append(i)
        except Exception as e:
            yield json.dumps({"status": "error", "message": f"분석 단계 오류: {str(e)}"}) + "\n"
            return 
        
        try:  
            if toxic_indices:
                for idx, list_idx in enumerate(toxic_indices):
                    yield json.dumps({"status": "progress","message": f"위험 조항({processed_results[list_idx]['id']})에 대한 개선안 생성 중..."}) + "\n"
                    
                    # 해당 결과 가져오기
                    target_result = processed_results[list_idx]
                    
                    # 개선안 생성 호출
                    try:
                        suggestion = detector.generate_easy_suggestion(target_result)
                        processed_results[list_idx]['suggestion'] = suggestion
                    except Exception as e:
                        processed_results[list_idx]['suggestion'] = "개선안 생성 실패"
            
            
            yield json.dumps({"status": "complete", "results": processed_results}) + "\n"
               
        except Exception as e:
            yield json.dumps({"status": "error", "message": f"개선안 생성 오류: {str(e)}"}) + "\n"
            return 
        
            

        # status: complete와 함께 결과 데이터 전송
        

    # StreamingResponse로 감싸서 반환 (media_type 중요)
    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
