import os
from dotenv import load_dotenv
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.g_eval import Rubric

# [Import]
from llm_service import LLM_gemini
from law.legal_context import LawContextManager
from law.precedent_context import PrecedentContextManager

#load_dotenv()

# --- 1. DeepEval용 Gemini 어댑터 ---
class GeminiDeepEvalAdapter(DeepEvalBaseLLM):
    def __init__(self, llm_service: LLM_gemini):
        self.llm_service = llm_service
        self.model_name = llm_service.model_name

    def load_model(self):
        return self.llm_service.client

    def generate(self, prompt: str) -> str:
        response = self.llm_service.generate(prompt)
        return response.text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# --- 2. 독소조항 판별기 클래스 ---
class ToxicClauseDetector:
    def __init__(self, api_key=None):
        print("🛡️ ToxicClauseDetector (Pro Model) 초기화 중...")
        
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API Key가 없습니다.")

        
        self.llm_service = LLM_gemini(gemini_api_key=api_key, model="gemini-2.5-flash")
        self.evaluator_llm = GeminiDeepEvalAdapter(self.llm_service)
        
        # DB 매니저
        self.law_manager = LawContextManager()
        self.precedent_manager = PrecedentContextManager()
        self.law_manager.initialize_database()
        self.precedent_manager.initialize_database()

        # [핵심 변경 2] 프롬프트 전략 변경: "안전 점수" -> "위험 점수(Toxicity Score)"
        # 점수가 높을수록 '위험(Toxic)'한 것으로 기준을 뒤집습니다.
        self.toxic_criteria = """
        당신은 근로자 보호를 최우선으로 하는 '근로계약서 감사관'입니다.
        입력된 조항이 [관련 법령/판례]를 위반하거나, 근로자에게 불공정한 '독소조항'인지 평가하세요.
        
        [필수 확인 사항]
        1. 퇴직금/주휴수당 포기 각서가 포함되어 있는가? (명백한 불법)
        2. '민형사상 이의 제기 금지' 같은 과도한 권리 제한이 있는가?
        3. 손해배상액을 미리 정해두었는가? (위약 예정 금지 위반)
        4. 해고 예고 없이 즉시 해고가 가능하다고 명시했는가?
        """

        self.rubric = [
            Rubric(score_range=(0, 2), expected_outcome="완벽하게 공정하고 법적인 문제가 전혀 없는 조항."),
            Rubric(score_range=(3, 5), expected_outcome="법적 효력은 있으나 근로자에게 다소 불리하거나 모호한 표현이 있음."),
            Rubric(score_range=(6, 8), expected_outcome="근로기준법 위반 소지가 높거나, 근로자에게 부당한 의무를 부과함."),
            Rubric(score_range=(9, 10), expected_outcome="명백한 불법(위약 예정, 퇴직금 포기 등)이며 절대 용납될 수 없는 독소조항."),
        ]

        self.evaluation_steps = [
            "조항의 핵심 의도(임금 삭감, 해고 용이성, 책임 전가 등)를 파악한다.",
            "[관련 법령]이 제공되지 않았더라도, 당신의 일반적인 법률 지식을 동원하여 위법성을 판단한다.",
            "특히 '퇴직금 포기', '손해배상 예정', '강제 근로' 관련 키워드가 있으면 즉시 최고 위험 점수(10점)를 부여한다.",
            "법적 근거가 확실하지 않으면 근로자에게 불리한 쪽으로 해석하여 점수를 매긴다."
        ]

    def _retrieve_context(self, clause_text):
        # 1. 법령 검색
        laws = self.law_manager.search_relevant_laws(clause_text, k=2)
        law_text = "\n".join(laws) if laws else "관련 법령 검색 결과 없음 (일반 법률 지식으로 판단 요망)"

        # 2. 판례 검색
        precedents = self.precedent_manager.search_relevant_precedents(clause_text, k=1)
        precedent_text = precedents[0] if precedents else "관련 판례 검색 결과 없음"

        return f"=== [관련 법령] ===\n{law_text}\n\n=== [관련 판례] ===\n{precedent_text}"

    def detect(self, clause_text):
        # print(f"🕵️ 조항 분석 중: {clause_text[:30]}...")
        
        retrieved_context = self._retrieve_context(clause_text)
        
        # G-Eval 평가
        toxic_metric = GEval(
            name="Toxicity Score", # 이름 변경
            criteria=self.toxic_criteria,
            rubric=self.rubric,
            evaluation_steps=self.evaluation_steps,
            model=self.evaluator_llm, 
            threshold=5, # 5점 이상이면 독소조항으로 간주
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
        )

        test_case = LLMTestCase(
            input=clause_text,
            actual_output="평가 대상",
            retrieval_context=[retrieved_context]
        )

        toxic_metric.measure(test_case)
        
        # [핵심 변경 3] 점수 해석 로직 단순화
        # 이제 점수(0~10)가 곧 위험도입니다. 뒤집을 필요가 없습니다.
        risk_score = toxic_metric.score # 0~10점 (DeepEval 버전에 따라 0~1일 수도 있음, 아래 보정)
        
        # DeepEval이 0~1 사이 값을 리턴하는 경우 10을 곱해줌
        if risk_score <= 1.0:
            risk_score *= 10
            
        # 4점 이상이면 독소조항 (기준 강화)
        is_toxic = risk_score >= 4.0
        
        # 디버깅용 출력 (터미널에서 확인 가능)
        print(f"[{'🚨위험' if is_toxic else '✅안전'}] 점수: {risk_score} | 내용: {clause_text[:20]}...")

        return {
            "clause": clause_text,
            "is_toxic": is_toxic,
            "risk_score": round(risk_score, 1),
            "reason": toxic_metric.reason,
            "context_used": retrieved_context
        }

    def generate_easy_suggestion(self, detection_result):
        if not detection_result['is_toxic']:
            return "✅ **안전한 조항입니다.**"

        prompt = f"""
        당신은 근로자 편인 법률 전문가입니다. 다음 독소조항을 분석하세요.
        
        [원문]: {detection_result['clause']}
        [이유]: {detection_result['reason']}
        [근거]: {detection_result['context_used']}

        다음 두 가지를 마크다운으로 작성:
        1. **⚠️ 쉬운 해석**: 초등학생도 이해하게 2문장 요약.
        2. **💡 수정 제안**: 법에 맞는 공정한 조항 예시.
        """
        response = self.llm_service.generate(prompt)
        return response.text