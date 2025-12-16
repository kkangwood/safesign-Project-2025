import os
import time
from typing import List, Dict, Union
from dotenv import load_dotenv

# DeepEval Imports
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.g_eval import Rubric
from deepeval.evaluate import AsyncConfig

from llm_service import LLM_gemini
from law.legal_context import LawContextManager
from law.precedent_context import PrecedentContextManager

# --- 1. DeepEval용 Gemini 어댑터 ---
class GeminiDeepEvalAdapter(DeepEvalBaseLLM):
    def __init__(self, llm_service: LLM_gemini):
        self.llm_service = llm_service
        self.model_name = llm_service.model_name

    def load_model(self):
        return self.llm_service.client

    def generate(self, prompt: str) -> str:
        response = self.llm_service.generate(prompt)
        return response.text if hasattr(response, 'text') else str(response)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# --- 2. 독소조항 판별기 클래스 ---
class ToxicClauseDetector:
    def __init__(self, api_key=None):
        print("🛡️ ToxicClauseDetector (Parallel) 초기화 중...")
        
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        
        self.llm_service = LLM_gemini(gemini_api_key=api_key, model="gemini-2.5-flash-lite")
        self.evaluator_llm = GeminiDeepEvalAdapter(self.llm_service)
        
        # DB 매니저
        self.law_manager = LawContextManager()
        self.precedent_manager = PrecedentContextManager()
        self.law_manager.initialize_database()
        self.precedent_manager.initialize_database()

        # [User Original Prompt & Logic] - 수정하지 않음
        self.toxic_criteria = """
       당신은 근로계약서의 [법적 효력], [공정성], [명확성]을 심사하는 전문 노무 AI입니다.
        아래 3가지 핵심 기준에 따라 조항을 분석하고 점수를 매기세요.

        [분석 기준 3대 원칙]
        1. 법적 효력 (Legality) - [치명적/Red Zone]
           - 근로기준법 등 강행법규를 위반하는가?
           - 예: 최저임금 미달, 퇴직금 포기 각서, 위약금 예정(손해배상액 명시), 해고 예고 위반.
           - 판단: 위반 시 무조건 9~10점 부여.

        2. 공정성 (Fairness) - [위험/Orange Zone]
           - 사용자(회사)에게만 유리하고 근로자에게 과도한 의무를 부과하는가?
           - 예: "모든 손해를 배상한다(포괄적 배상)", "퇴사 시 후임자를 구해야 한다", "사내 규정 위반 시 무조건 징계".
           - 판단: 불법은 아니더라도 근로자가 억울할 소지가 크면 6~8점 부여.

        3. 명확성 (Clarity) - [주의/Yellow Zone]
           - 자의적 해석이 가능한 모호한 표현이 있는가?
           - 예: "회사가 필요하다고 인정하는 경우", "관례에 따른다", "기타 갑이 정하는 업무".
           - 판단: 문구가 모호하여 분쟁 가능성이 있으면 3~5점 부여.
        """
        
        self.rubric = [
            Rubric(score_range=(0, 2), expected_outcome="3대 원칙(효력, 공정성, 명확성)을 모두 충족하는 완벽한 조항."),
            Rubric(score_range=(3, 5), expected_outcome="[명확성 부족] - 법적으로 문제는 없으나, 표현이 모호하여 회사의 자의적 해석이 우려됨."),
            Rubric(score_range=(6, 8), expected_outcome="[공정성 결여] - 불법 직전의 수준. 근로자에게 일방적으로 불리하거나 입증 책임을 전가함."),
            Rubric(score_range=(9, 10), expected_outcome="[법적 효력 없음] - 근로기준법 강행규정 위반으로 해당 조항 자체가 무효임."),
        ]

        self.evaluation_steps = [
            "1단계 [의도 파악]: 조항의 핵심 의도(임금 삭감, 해고 용이성, 책임 전가 등)를 먼저 파악하고, 일반적인 법률 지식을 로딩한다.",
            "2단계 [Legality/치명적]: 근로기준법 강행규정 위반 여부를 최우선 확인한다. 특히 '퇴직금 포기', '손해배상액 예정', '강제 근로' 키워드가 있으면 즉시 10점을 부여한다.",
            "3단계 [Fairness/위험]: 불법이 아니라면, 권리와 의무의 균형을 본다. 회사에만 유리하거나 근로자에게 과도한 의무를 부과하면 6~8점을 부여한다.",
            "4단계 [Clarity/주의]: 내용이 공정해 보여도, '기타', '상당한' 등 자의적 해석이 가능한 모호한 단어가 있다면 3~5점을 부여한다.",
            "5단계 [종합 판단]: 위 단계들을 거쳐 점수를 매기되, 법적 근거가 확실하지 않은 회색지대라면 근로자에게 불리한 쪽(보수적)으로 해석하여 최종 점수를 확정한다.",

        ]
        
        # Metric 객체 초기화 (재사용)
        self.toxic_metric = GEval(
            name="Toxicity Score",
            criteria=self.toxic_criteria,
            rubric=self.rubric,
            evaluation_steps=self.evaluation_steps,
            model=self.evaluator_llm,
            threshold=5, 
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
        )

    def _retrieve_context(self, clause_text):
        laws = self.law_manager.search_relevant_laws(clause_text, k=2)
        law_text = "\n".join(laws) if laws else "관련 법령 검색 결과 없음 (일반 법률 지식으로 판단 요망)"

        precedents = self.precedent_manager.search_relevant_precedents(clause_text, k=1)
        precedent_text = precedents[0] if precedents else "관련 판례 검색 결과 없음"

        return f"=== [관련 법령] ===\n{law_text}\n\n=== [관련 판례] ===\n{precedent_text}"

    # 함수명 'detect' 유지 (Input: List[str]로 변경됨)
    def detect(self, clause_texts: List[str], max_concurrent: int = 5) -> List[Dict]:
        """
        [병렬 처리] 여러 조항을 리스트로 받아 DeepEval evaluate 함수로 한 번에 처리.
        """
        print(f"🚀 총 {len(clause_texts)}개 조항에 대한 병렬 평가 시작...")
        
        test_cases = []
        original_map = {} # 결과 매핑용

        # 1. Test Case 생성 (Retrieval 수행)
        for text in clause_texts:
            retrieved_context = self._retrieve_context(text)
            test_case = LLMTestCase(
                input=text,
                actual_output="평가 대상",
                retrieval_context=[retrieved_context]
            )
            test_cases.append(test_case)
            original_map[text] = retrieved_context
        # 2. 병렬 평가 실행 (evaluate)
        eval_results = evaluate(
            test_cases=test_cases,
            metrics=[self.toxic_metric],
            async_config=AsyncConfig(max_concurrent=max_concurrent), # 병렬 처리 개수
        )

        # 3. 결과 포맷팅 (수정된 로직)
        formatted_results = []
        
        # [핵심] eval_results 객체에서 진짜 결과 리스트(.test_results)를 꺼냅니다.
        if hasattr(eval_results, 'test_results'):
            actual_test_results = eval_results.test_results
        elif isinstance(eval_results, list):
            actual_test_results = eval_results
        else:
            # 혹시 모를 상황 대비 (딕셔너리 등)
            print("⚠️ 결과 형식이 예상과 다릅니다. Raw Data를 확인하세요.")
            actual_test_results = []

        for result in actual_test_results:
            # result는 이제 'TestResult' 객체입니다.
            
            # 메트릭 데이터가 존재하는지 방어적 코딩
            if not result.metrics_data:
                continue

            # 우리는 metric을 하나만 넣었으므로 0번째 인덱스를 가져옵니다.
            metric_data = result.metrics_data[0] 
            clause_text = result.input
            
            # 점수 계산 (User Logic 유지)
            # score는 보통 0.0~1.0 사이로 나오므로 10배 해줍니다.
            risk_score = metric_data.score
            if risk_score <= 1.0:
                risk_score *= 10
            
            is_toxic = risk_score >= 4.0

            formatted_results.append({
                "clause": clause_text,
                "is_toxic": is_toxic,
                "risk_score": round(risk_score, 1),
                "reason": metric_data.reason,
                "context_used": original_map.get(clause_text, "")
            })

        return formatted_results

    def generate_easy_suggestion(self, detection_result):
        if not detection_result['is_toxic']:
            return "✅ **안전한 조항입니다.**"

        prompt = f"""
        당신은 근로자 편인 법률 전문가입니다. 다음 독소조항을 분석하세요.
        
        [원문]: {detection_result['clause']}
        [이유]: {detection_result['reason']}
        [근거]: {detection_result['context_used']}

        다음 두 가지를 마크다운으로 작성:
        1. **⚠️ 쉬운 해석**: 근로자가 이해하기 쉽게 1~2문장으로 '왜 위험한지' 설명 및 요약.
        2. **💡 수정 제안**: 법에 맞는 공정한 조항 예시.
        """
        response = self.llm_service.generate(prompt)
        return response.text if hasattr(response, 'text') else str(response)

# --- 3. 테스트 코드 ---
if __name__ == "__main__":
    # os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"
    
    detector = ToxicClauseDetector()

    # 테스트할 조항들 (리스트 형태)
    test_clauses = [
        "퇴사 시 후임자를 구하지 못하면 손해배상을 청구한다.", # 독소조항 (높은 점수 예상)
        "근로시간은 09시부터 18시까지로 한다.",              # 정상조항 (낮은 점수 예상)
        "수습기간 중에는 급여의 50%만 지급한다."               # 독소조항 (최저임금법 위반 가능성)
    ]

    # 병렬 실행 (함수명 detect 유지)
    results = detector.detect(test_clauses, max_concurrent=3)

    print("\n" + "="*50)
    for res in results:
        status = "🚨위험" if res['is_toxic'] else "✅안전"
        print(f"[{status}] 점수: {res['risk_score']} | 조항: {res['clause'][:30]}...")
        print(f"   이유: {res['reason']}")
        print("-" * 50)