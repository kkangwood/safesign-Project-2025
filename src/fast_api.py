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
def process_single_clause(detector, clause, index):
    """단위 작업: 조항 하나 분석"""
    try:
        detection = detector.detect(clause)
        
        suggestion = ""
        if detection['is_toxic']:
            suggestion = detector.generate_easy_suggestion(detection)
            
        return {
            "id": index + 1,
            "clause": clause,
            "is_toxic": detection['is_toxic'],
            "score": detection['risk_score'],
            "reason": detection['reason'],
            "context": detection['context_used'],
            "suggestion": suggestion,
            "status": "success"
        }
    except Exception as e:
        return {
            "id": index + 1,
            "clause": clause,
            "error": str(e),
            "status": "error"
        }
class AnalyzeRequest(BaseModel):
    api_key: str
    text: str
@app.post("/analyze")
async def analyze_contract(request: AnalyzeRequest):
    # 제너레이터 함수: 데이터를 조금씩 나누어 보냅니다.
    async def event_stream():
        try:
            detector = ToxicClauseDetector(gemini_api=request.api_key)
            results = []
            chunks  = parse_text_to_chunks(request.text)

            yield json.dumps({"status": "progress", "current": 0, "total": len(chunks), "message": "분석 시작..."}) + "\n"
            for i, clause in enumerate(chunks):
                res = process_single_clause(detector, clause, i)
                
                results.append(res)
                if i%5 == 0 or (i+1 == len(chunks)):
                    yield json.dumps({"status": "progress", "current": i+1,"total": len(chunks) , "message": "AI가 독소 조항을 판별 중입니다..."}) + "\n"
            
            await asyncio.sleep(0.01)    

            # status: complete와 함께 결과 데이터 전송
            yield json.dumps({"status": "complete", "results": results}) + "\n"

        except Exception as e:
            # 에러 발생 시 에러 메시지 전송
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    # StreamingResponse로 감싸서 반환 (media_type 중요)
    return StreamingResponse(event_stream(), media_type="application/x-ndjson")