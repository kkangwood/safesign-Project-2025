from google import genai
from google.genai import types

# 1. gemini 객체 만들기
class LLM_gemini():
    # 사용자의 API Key와 사용할 모델을 입력
    def __init__(self, gemini_api_key, model):
        self.GEMINI_API_KEY = gemini_api_key
        self.model_name = model
        self.client = genai.Client(api_key=self.GEMINI_API_KEY)

    # pdf를 text로 추출하는 함수
    # 아직까지는 pdf를 text로 추출할 때만 gemini를 사용하기 때문에 client를 함수 내부에서 생성했다.
    # 입력값은 pdf_bytes = st.file_uploader().read()
    def pdf_to_text(self, pdf_file_bytes):
        extraction_prompt = """
        당신은 법률 문서 텍스트 추출 전문가입니다. 
        제공된 PDF 파일의 내용을 읽고, 아래 [출력 양식]에 맞춰 텍스트만 정확하게 추출하세요.
        
        [규칙]
        1. 절대 내용을 요약하거나 생략하지 마십시오. (Verbatim Extraction)
        2. 조항의 제목과 본문은 줄바꿈으로 구분하십시오.
        3. 각 조항 사이에는 빈 줄을 추가하십시오.
        4. 서문(계약 당사자 정의 등)이 있다면 맨 위에 적어주십시오.
        5. 번호를 매길 때 "제(숫자)조" 형태로 적으시오.

        [출력 양식 예시]
        제1조 (목적)
        본 계약은 사용자 (주)악덕상사(이하 "갑")와...

        제2조 (임금)
        1. 월 급여는...
        2. 위 급여에는...
        """
        
        response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    genai.types.Part.from_bytes(data=pdf_file_bytes, mime_type="application/pdf"),
                    extraction_prompt 
                ]
            )
        return response.text
    
    # prompt에 맞는 텍스트를 출력하는 함수
    # toxic_detector.py에서 필요하다.
    def generate(self,prompt):
        # [수정] config에 temperature=0.0 추가
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0  # 0.0으로 설정하면 가장 논리적이고 일관된 답변을 줍니다.
            )
        )
        return response
    

