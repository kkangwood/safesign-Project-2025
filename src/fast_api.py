from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from starlette.middleware.cors import CORSMiddleware
from llm_service import LLM_gemini


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