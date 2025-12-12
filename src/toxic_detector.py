import os
import json
import re
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI (염준화)
from llm_service import LLM_gemini

# [추가] 실행을 위한 필수 라이브러리 및 임시 클래스 정의
# ---------------------------------------------------------------------------
# 1. RAG 파이프라인이 아직 없거나 에러가 날 경우를 대비한 안전 장치
try:
    from .rag_pipeline import search_precedents
except ImportError:
    def search_precedents(text, k=1): return ["(판례 검색 모듈이 아직 구현되지 않음)"]

# 2. DeepEval 인터페이스 호환을 위한 Gemini Wrapper
# llm_service의 LLM_gemini로 통합 (염준화)

# class GeminiWrapper:
#     def __init__(self):
#         load_dotenv()
#         api_key = os.getenv("GEMINI_API_KEY")
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-2.5-flash",
#             temperature=0.0,
#             google_api_key=api_key
#         )
    
#     def generate(self, prompt):
#         return self.llm.invoke(prompt).content

# 3. LawManager가 구현되지 않았을 때를 위한 Mock Class
class MockLawManager:
    def initialize_database(self):
        pass # 실제 DB 연결 로직이 들어갈 곳
    
    def search_relevent_laws(self, text, k=2):
        # 원본 코드의 오타(relevent)를 그대로 지원하기 위한 메서드
        return ["근로기준법 제20조 (위약 예정의 금지): 사용자는 근로계약 불이행에 대한 위약금 또는 손해배상액을 예정하는 계약을 체결하지 못한다."]
# ---------------------------------------------------------------------------

from deepeval.metrics.g_eval import Rubric
# from .rag_pipeline import search_precedents (상단 try-except로 이동하여 처리함)

class ToxicClauseDetector():
    # 사용자 gemini_api를 가져올 수 있게 수정(염준화)
    def __init__(self,gemini_api):
        # 독소조항 판병하는 llm 모델, DeepEval용 Wrapper 써야함
        # 
        self.evaluator_llm = LLM_gemini(gemini_api,"gemini-2.5-flash")
        
        # 법령 정보를 관리하는 인스턴스, DB 초기화 및 법령 검색 기능
        self.law_manager = MockLawManager()
        self.law_manager.initialize_database()
        
        # 독소조항 판별에 쓰일 평가기준
        self.toxic_criteria = """
        1. 법적 효력 (Legality): 근로기준법 등 강행법규를 위반하는가?
        2. 공정성 (Fairness): 사용자에게만 유리하고 근로자에게 과도한 의무를 부과하는가?
        3. 명확성 (Clarity): 자의적 해석이 가능한 모호한 표현이 있는가?
        """
        
        # 독소조항 판별에 쓰일 rubric
        self.rubric = [
            "0점: 법적, 윤리적으로 완벽하게 안전함",
            "50점: 근로자에게 불리하거나 다소 모호하여 주의가 필요함",
            "100점: 근로기준법 위반 소지가 높거나 명백한 독소조항임"
        ]
        
        # 독소조항 판별에 쓰일 CoT
        self.evalutation_step = [
            "입력된 조항과 검색된 법령/판례를 비교 분석한다.",
            "평가 기준(법적 효력, 공정성, 명확성)에 따라 위반 요소를 식별한다.",
            "위반 정도에 따라 루브릭 점수를 산정한다.",
            "최종 결과를 JSON 포맷으로 생성한다."
        ]

    # 법령 및 판례 검색하는 함수
    def _retrieve_context(self, clause_text):
        # 1. 법령 검색
        laws = self.law_manager.search_relevent_laws(clause_text, k = 2)
        law_text = "\n".join(laws) if laws else "관련 법령 없음"

        # 2. 판례 검색
        precedents = search_precedents(clause_text, k = 1)
        precedent_text = precedents[0] if precedents else "관련 판례 없음"

        return f"[관련 법령]\n{law_text}\n\n[관련 판례]\n{precedent_text}"
    
    # DeepEval 라이브러리를 써서 독소조항 여부를 판별하는 함수
    def detect(self, clause_text):
        # Context 검색
        context = self._retrieve_context(clause_text)
        
        # 프롬프트 구성
        prompt = f"""
        당신은 전문 법률 AI입니다. 아래 정보를 바탕으로 근로계약서 조항을 분석하세요.

        [평가 단계]
        {chr(10).join(self.evalutation_step)}

        [평가 기준]
        {self.toxic_criteria}

        [채점 루브릭]
        {chr(10).join(self.rubric)}

        [입력 조항]
        "{clause_text}"

        [참고 법령 및 판례]
        {context}

        [출력 형식]
        오직 JSON 형식으로만 응답하세요:
        {{
            "is_toxic": true (점수가 50점 이상이면 true),
            "risk_score": 0~100 정수,
            "reason": "판단 근거 한 줄 요약",
            "suggestion": "수정 제안 (문제가 없으면 '수정 불필요')"
        }}
        """

        try:
            # LLM 호출
            response_text = self.evaluator_llm.generate(prompt)
            
            # JSON 파싱 (Markdown 코드 블록 제거 처리)
            cleaned_text = re.sub(r'```json|```', '', response_text).strip()
            result = json.loads(cleaned_text)
            
            # 반환값 구성
            is_toxic = result.get("is_toxic", False)
            risk_score = result.get("risk_score", 0)
            reason = result.get("reason", "분석 불가")
            retrieved_context = context

            return {
                "clause": clause_text,
                "is_toxic": is_toxic,
                "risk_score": risk_score, # 메인 UI 호환용 추가
                "reason": reason,
                "suggestion": result.get("suggestion", ""), # 메인 UI 호환용 추가
                "context_used": retrieved_context
            }
            
        except Exception as e:
            # 에러 발생 시 기본값 반환
            return {
                "clause": clause_text,
                "is_toxic": False,
                "risk_score": 0,
                "reason": f"AI 분석 중 오류 발생: {str(e)}",
                "suggestion": "다시 시도해주세요.",
                "context_used": context
            }
    
    # 
    def generate_easy_suggestion(self, detection_result):
        if not detection_result.get('is_toxic'):
            return "✅ 법적으로 문제가 없는 안전한 조항입니다."

        prompt = f"""
        다음은 근로계약서 독소조항에 대한 분석 결과입니다.
        근로자가 이해하기 쉽게 1~2문장으로 '왜 위험한지' 설명하고,
        어떻게 고쳐야 하는지 친절하게 알려주세요.

        [분석 결과]
        {detection_result.get('reason')}
        
        [수정 제안]
        {detection_result.get('suggestion')}
        """
        return self.evaluator_llm.generate(prompt)