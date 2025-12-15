# src/rag_pipeline.py
import json
import re
import os
from dotenv import load_dotenv

# [Import]
from toxic_detector import ToxicClauseDetector
from llm_service import LLM_gemini
from law.legal_context import LawContextManager
from law.precedent_context import PrecedentContextManager

load_dotenv()

class RagPipeline:
    """
    RAG 파이프라인: 유해성 검사 -> 검색 -> 답변 생성 -> Faithfulness 검증 루프
    """
    def __init__(self):
        print("⚙️ RAG 파이프라인 초기화 중...")
        
        api_key = os.getenv("GEMINI_API_KEY")
        self.llm = LLM_gemini(gemini_api_key=api_key, model="gemini-1.5-flash")
        
        # Detector 재사용 (이미 DB 매니저를 가지고 있음)
        self.toxic_detector = ToxicClauseDetector()
        
        # DB 매니저 접근을 위해 Detector 내부 객체 참조
        self.law_manager = self.toxic_detector.law_manager
        self.precedent_manager = self.toxic_detector.precedent_manager

        self.MAX_RETRIES = 2
        self.TARGET_SCORE = 75

    def run(self, user_query: str):
        """
        사용자 질문에 대해 법률적 답변을 생성하는 전체 파이프라인
        """
        print(f"\n🚀 [Pipeline Start] 질문: {user_query}")

        # 1. 검색 (Retrieval)
        law_docs = self.law_manager.search_relevant_laws(user_query, k=2)
        prec_docs = self.precedent_manager.search_relevant_precedents(user_query, k=2)

        if not law_docs and not prec_docs:
            return {
                "answer": "관련된 법령이나 판례 정보를 찾을 수 없습니다.",
                "sources": [],
                "score": 0
            }
        
        context_text = self._format_context(law_docs, prec_docs)

        # 2. 생성 및 검증 루프 (Generation & Loop)
        current_answer = ""
        current_score = 0
        retry_count = 0
        feedback = ""

        while retry_count <= self.MAX_RETRIES:
            print(f"📝 [Attempt {retry_count + 1}] 답변 생성 중...")
            
            # (1) 답변 생성
            current_answer = self._generate_answer(user_query, context_text, feedback)

            # (2) Faithfulness 평가
            eval_result = self._evaluate_faithfulness(user_query, current_answer, context_text)
            current_score = eval_result.get('score', 0)
            reason = eval_result.get('reason', '평가 불가')
            
            print(f"   👉 점수: {current_score}점 | 이유: {reason}")

            if current_score >= self.TARGET_SCORE:
                break
            else:
                feedback = f"점수 미달({current_score}점). 이유: {reason}. 근거 자료에만 기반하여 다시 작성하세요."
                retry_count += 1

        final_sources = law_docs + prec_docs
        
        if current_score < self.TARGET_SCORE:
            current_answer = f"[주의: 근거 불충분 (신뢰도: {current_score}%)]\n{current_answer}"

        return {
            "answer": current_answer,
            "sources": final_sources,
            "score": current_score
        }

    def _format_context(self, laws, precedents):
        formatted = ""
        if laws:
            formatted += "=== [관련 법령] ===\n" + "\n".join([f"{i+1}. {txt}" for i, txt in enumerate(laws)]) + "\n\n"
        if precedents:
            formatted += "=== [관련 판례] ===\n" + "\n".join([f"{i+1}. {txt}" for i, txt in enumerate(precedents)])
        return formatted

    def _generate_answer(self, query, context, feedback=""):
        system_role = "당신은 대한민국 법률 AI입니다. 반드시 [참고 자료]에 기반하여 답변하세요."
        prompt = f"{system_role}\n\n[참고 자료]\n{context}\n\n[질문]\n{query}"
        if feedback:
            prompt += f"\n\n[수정 지시]\n{feedback}"
        
        response = self.llm.generate(prompt)
        return response.text

    def _evaluate_faithfulness(self, query, answer, context):
        prompt = f"""
        당신은 Fact Checker입니다. [참고 자료]를 바탕으로 [AI 답변]이 사실에 부합하는지 0~100점으로 평가하세요.
        결과는 JSON으로만 출력하세요: {{"score": 85, "reason": "..."}}

        [참고 자료]
        {context}
        [AI 답변]
        {answer}
        """
        try:
            response = self.llm.generate(prompt)
            clean_json = re.sub(r'```json|```', '', response.text).strip()
            # 중괄호 추출
            start = clean_json.find('{')
            end = clean_json.rfind('}') + 1
            return json.loads(clean_json[start:end])
        except:
            return {"score": 50, "reason": "Evaluation Error"}