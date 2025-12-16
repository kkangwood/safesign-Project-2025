import streamlit as st
import re
import os
import time
from dotenv import load_dotenv

# [Import] src 폴더 내의 모듈을 불러옵니다.
# 실제 파일 경로에 맞게 수정 필요 (예: from src.toxic_detector import ...)
from toxic_detector import ToxicClauseDetector
from llm_service import LLM_gemini

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="근로계약서 독소조항 판별기 (Parallel)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 헬퍼 함수들 (PDF 파싱 & 더미 데이터) ---

def extract_text_from_pdf(pdf_file, api_key, model_name): 
    """PDF 파일에서 텍스트를 추출하는 함수"""
    try:
        pdf_file_bytes = pdf_file.read()   
        gemini = LLM_gemini(gemini_api_key=api_key, model=model_name)
        result = gemini.pdf_to_text(pdf_file_bytes)
        return result
    except Exception as e:
        st.error(f"PDF 처리 중 오류 발생: {e}")
        return None

def get_dummy_contract_text():
    """테스트용 가상 근로계약서 텍스트"""
    return """
제1조 (목적)
본 계약은 사용자 (주)악덕상사(이하 "갑")와 근로자 홍길동(이하 "을")의 근로조건을 정함을 목적으로 한다.

제2조 (근로장소 및 업무)
"을"은 "갑"의 본사 및 "갑"이 지정하는 장소에서 소프트웨어 개발 업무를 수행한다.

제3조 (근로시간)
1. 근로시간은 09:00부터 18:00까지로 한다 (휴게시간 1시간 포함).
2. "갑"은 업무상 필요한 경우 "을"에게 연장, 야간 및 휴일근로를 명할 수 있으며 "을"은 이에 동의한 것으로 간주한다.

제4조 (임금)
1. 월 급여는 2,500,000원으로 한다.
2. 위 급여에는 식대, 교통비 및 법정 제수당(연장, 야간, 휴일근로수당 등)이 모두 포함된 포괄임금으로 산정하며, "을"은 추가적인 수당을 청구하지 않는다.

제5조 (퇴직금)
"을"이 입사 후 1년 미만에 퇴사하는 경우, 수습기간 동안의 교육비 및 손해배상 명목으로 퇴직금은 지급하지 아니한다.

제6조 (계약해지)
"을"이 무단결근 3일 이상 지속하거나 업무 능력이 현저히 부족하다고 판단될 경우 "갑"은 즉시 계약을 해지할 수 있다.

제7조 (손해배상)
"을"이 계약기간 중 퇴사하여 "갑"에게 손해를 입힌 경우, "을"은 "갑"에게 일금 1,000만원을 배상하여야 한다.
"""

def parse_text_to_chunks(text):
    """텍스트를 '제N조' 기준으로 자르는 파서"""
    if not text:
        return []
    split_pattern = r'(?=\n\s*제\s*\d+\s*조)'
    chunks = re.split(split_pattern, text)
    # 공백 제거 및 유효한 조항만 필터링 (너무 짧은 문장은 제외)
    clean_chunks = [c.strip() for c in chunks if len(c.strip()) > 10]
    return clean_chunks

# --- 3. 메인 어플리케이션 --- 
def main():
    # 사이드바 설정
    with st.sidebar:
        st.title("⚖️ Contract Guardian")
        st.caption("Parallel Processing Edition")
        st.markdown("---")
        
        load_dotenv()
        env_key = os.getenv("GEMINI_API_KEY")
        
        api_key_input = st.text_input(
            "Gemini API Key", 
            value=env_key if env_key else "", 
            type="password"
        )
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input

        st.info("💡 PDF 파일을 올리면 해당 내용을, 올리지 않으면 예시 데이터를 분석합니다.")

    # 메인 화면
    st.title("📄 근로계약서 독소조항 판별기")
    st.markdown("계약서를 업로드하면 AI가 **병렬 처리(Parallel Processing)**를 통해 신속하게 독소조항을 찾아냅니다.")

    # 파일 업로드 및 텍스트 로딩
    uploaded_file = st.file_uploader("근로계약서 PDF 업로드 (선택사항)", type=["pdf"])
    
    contract_content = ""
    
    if uploaded_file is not None:
        with st.spinner("PDF에서 텍스트를 추출하는 중..."):
            extracted_text = extract_text_from_pdf(uploaded_file, api_key_input, 'gemini-1.5-flash')
            if extracted_text:
                contract_content = extracted_text
                st.success("PDF 텍스트 추출 완료!")
            else:
                contract_content = get_dummy_contract_text()
                st.warning("PDF 텍스트 추출 실패. 예시 데이터를 사용합니다.")
    else:
        contract_content = get_dummy_contract_text()

    # 텍스트 에디터
    final_text = st.text_area("계약서 내용 확인 및 수정", value=contract_content, height=300)

    # API 키 체크
    if not os.environ.get("GEMINI_API_KEY"):
        st.warning("⚠️ 왼쪽 사이드바에 API Key를 입력해주세요.")
        return

    # [분석 버튼]
    if st.button("🚀 독소조항 고속 분석 시작", use_container_width=True):
        
        # 1. Parsing
        chunks = parse_text_to_chunks(final_text)
        
        if not chunks:
            st.error("분석할 조항을 찾지 못했습니다. 텍스트에 '제N조' 형식이 포함되어 있는지 확인해주세요.")
            st.stop()

        # 2. Detector 초기화 (캐싱)
        @st.cache_resource
        def get_detector(key):
            return ToxicClauseDetector(key)
        
        with st.spinner("⚙️ AI 엔진 및 법률 DB 로딩 중..."):
            detector = get_detector(api_key_input)

        st.info(f"총 {len(chunks)}개의 조항을 병렬로 분석합니다. 잠시만 기다려주세요...")

        # --- [핵심 변경] 병렬 처리 실행 ---
        try:
            start_time = time.time()
            
            # DeepEval의 evaluate 함수가 내부적으로 병렬 처리를 수행합니다.
            # 루프를 돌리지 않고 리스트 전체를 넘깁니다.
            raw_results = detector.detect(chunks, max_concurrent=5)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
        except Exception as e:
            st.error(f"분석 중 치명적인 오류 발생: {e}")
            st.stop()

        # 3. 결과 후처리 (ID 부여 및 개선안 생성)
        processed_results = []
        toxic_indices = [] # 개선안 생성이 필요한 인덱스들
        
        # 3-1. 기본 결과 매핑
        for i, res in enumerate(raw_results):
            # detect 함수에서 나온 결과에 ID(조항 번호) 추가
            res['id'] = i + 1
            res['suggestion'] = "" # 초기화
            processed_results.append(res)
            
            if res['is_toxic']:
                toxic_indices.append(i)

        # 3-2. 개선안 생성 (위험한 조항만 순차/병렬 처리)
        # 평가는 빨라도 생성(Suggestion)은 시간이 걸리므로 진행상황을 보여줍니다.
        if toxic_indices:
            suggestion_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, list_idx in enumerate(toxic_indices):
                status_text.text(f"💡 위험 조항({processed_results[list_idx]['id']}조)에 대한 개선안을 생성 중입니다...")
                
                # 해당 결과 가져오기
                target_result = processed_results[list_idx]
                
                # 개선안 생성 호출
                try:
                    suggestion = detector.generate_easy_suggestion(target_result)
                    processed_results[list_idx]['suggestion'] = suggestion
                except Exception as e:
                    processed_results[list_idx]['suggestion'] = "개선안 생성 실패"
                
                suggestion_bar.progress((idx + 1) / len(toxic_indices))
            
            status_text.empty()
            suggestion_bar.empty()

        st.success(f"✅ 분석 완료! (소요 시간: {elapsed_time:.2f}초)")
        
        # 4. 결과 리포트 출력
        st.divider()
        
        # 요약 지표
        toxic_count = len(toxic_indices)
        col1, col2 = st.columns(2)
        col1.metric("분석된 조항", f"{len(chunks)}건")
        col2.metric("발견된 위험 조항", f"{toxic_count}건", delta="-주의" if toxic_count > 0 else "안전")

        # 상세 결과 탭
        tab1, tab2 = st.tabs(["🚨 위험 조항 리포트", "📑 전체 조항 보기"])
        
        with tab1:
            if toxic_count == 0:
                st.balloons()
                st.success("완벽합니다! 독소조항이 발견되지 않았습니다.")
            else:
                for res in processed_results:
                    if res.get('is_toxic'):
                        # 위험도에 따른 색상 구분 (선택사항)
                        risk_label = "치명적" if res['risk_score'] >= 9 else "위험"
                        
                        with st.expander(f"⚠️ [{risk_label}] 제{res['id']}조 (위험도: {res['risk_score']})", expanded=True):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.caption("❌ 원문 조항")
                                st.error(res['clause'])
                                st.markdown(f"**🔍 판단 근거:**\n{res['reason']}")
                            with c2:
                                st.caption("💡 AI 수정 제안")
                                if res['suggestion']:
                                    st.markdown(res['suggestion'])
                                else:
                                    st.info("개선안 생성 중...")
                                
                                with st.popover("📜 참고 법령/판례 보기"):
                                    st.text(res['context_used'])
        
        with tab2:
            st.caption("모든 조항에 대한 AI의 평가 결과입니다.")
            for res in processed_results:
                icon = "🔴" if res.get('is_toxic') else "🟢"
                score_badge = f"(점수: {res['risk_score']})"
                
                with st.expander(f"{icon} 제{res['id']}조 {score_badge}"):
                    st.code(res['clause'], language="text")
                    st.write(f"**판단:** {res['reason']}")

if __name__ == "__main__":
    main()