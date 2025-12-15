import os
import time
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # LangChain Deprecation 경고를 피하기 위한 새로운 임포트
from langchain_core.documents import Document

# --- 설정 ---
# ⭐️ DB 저장 경로 및 모델 정의
DB_PATH = "../data/faiss_precedent_db" 
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli" 
# ⭐️ Hugging Face 데이터셋 ID
DATASET_ID = "joonhok-exo-ai/korean_law_open_data_precedents" 
SAMPLE_SIZE = 1000 # 테스트/구축용 데이터 개수 (전체 사용 시 None)

class PrecedentContextManager:
    """
    Hugging Face 데이터셋을 기반으로 판례 벡터 DB를 구축하고 관리하는 클래스입니다.
    이 클래스는 DB 로드와 검색 기능을 통합 제공합니다.
    """
    def __init__(self):
        self.vectorstore = None
        # 임베딩 모델 객체 초기화 (DB 로드/구축 시 사용)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    def _fetch_and_parse_precedents(self):
        """
        Hugging Face 데이터셋에서 판례를 다운로드하고 LangChain Document 객체로 변환합니다.
        
        :return: 변환된 Document 객체 리스트
        """
        print(f"📥 판례 데이터셋 다운로드 중... ({DATASET_ID})")
        
        # 1. 데이터셋 로드 및 샘플링
        try:
            dataset = load_dataset(DATASET_ID, split="train") 
            
            if SAMPLE_SIZE and len(dataset) > SAMPLE_SIZE:
                dataset = dataset.select(range(SAMPLE_SIZE)) 
                print(f"    - (설정) 상위 {SAMPLE_SIZE}개만 벡터화합니다.")
                
        except Exception as e:
            print(f"❌ 데이터셋 로드 실패: {e}")
            return []
        
        # 2. Document 객체로 변환
        print("🔄 문서 객체(Document)로 변환 중...")
        documents = []

        for item in dataset:
            # 데이터셋 컬럼 매핑 및 내용 추출
            content = item.get('전문', '')
            summary = item.get('판결요지', '')
            case_name = item.get('사건명', '사건명 정보 없음')
            case_number = item.get('사건번호', 'N/A')

            # 검색 정확도를 위한 page_content 구성 (판시사항, 요지 등 중요 정보 강조)
            page_content = f"""
[사건번호] {case_number}
[사건명] {case_name}
[판결요지] {summary}
[전문] {content[:2000]}...
""".strip()
            
            metadata = {
                "case_name": case_name, 
                "source": "HuggingFace Precedent DB",
                "case_number": case_number
            }
            
            # 유효성 검사 (판례 요지가 충분히 긴 경우에만 포함)
            if len(summary) > 10: 
                 documents.append(Document(page_content=page_content, metadata=metadata))
        
        print(f"    - 변환된 유효 문서: {len(documents)}개")
        return documents

    def initialize_database(self):
        """
        로컬 DB 경로를 확인하여 기존 DB를 로드하거나, 없을 경우 신규 구축 후 저장합니다.
        """
        if self.vectorstore is not None:
            print("💡 판례 DB가 이미 로드되었습니다.")
            return

        # 1. 로컬 DB 파일 존재 확인 및 로드 시도
        if os.path.exists(DB_PATH) and os.path.isdir(DB_PATH):
            print(f"✅ [초기화] 기존 판례 DB 로드 중... (경로: {DB_PATH})")
            try:
                # FAISS 로드 (DB 구축 시 사용된 임베딩 모델 객체 전달)
                self.vectorstore = FAISS.load_local(
                    DB_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print(f"✅ [초기화] 판례 DB 로드 완료! (총 {len(self.vectorstore.docstore._dict)}건)")
                return
            except Exception as e:
                print(f"⚠️ 기존 DB 로드 실패: {e}. DB를 새로 구축합니다.")
        
        # 2. 신규 DB 구축
        print(f"📚 [초기화] 필수 판례 데이터 신규 구축을 시작합니다. (페이지당 {display}건, 최대 {max_pages} 페이지)")
        all_docs = []       # 수집된 모든 Document 객체 리스트
        precedent_ids = set() # 판례일련번호 중복 방지용 Set

        for query in self.target_queries:
            print(f"\n  🔍 '{query}' 검색 중...")
            page = 1
            
            # 검색 결과를 중복 없이 수집
            while page <= max_pages: 
                # 판례 목록 검색
                precedents, total_count = search_precedent_list(query, display=display, page=page)
                
                if not precedents:
                    break
                
                total_pages = (total_count + display - 1) // display
                print(f"  📥 페이지 {page}/{total_pages} ({len(precedents)}건) 판례 상세 다운로드 및 파싱...")

                for prec_info in precedents:
                    # ==========================================
                    # [수정됨] 데이터 타입 방어 코드 추가 구간
                    # ==========================================
                    
                    # 1. 문자열(String)이 잘못 들어온 경우 체크
                    if isinstance(prec_info, str):
                        print(f"⚠️ [경고] 예상치 못한 데이터 타입(str) 발견 -> 건너뜀. 내용: {prec_info}")
                        continue
                    
                    # 2. 딕셔너리가 아닌 경우 체크
                    if not isinstance(prec_info, dict):
                        print(f"⚠️ [경고] 딕셔너리가 아닌 데이터 타입({type(prec_info)}) 발견 -> 건너뜀.")
                        continue

                    # 3. 안전하게 .get() 호출
                    prec_id = prec_info.get("판례일련번호")
                    
                    if not prec_id or prec_id in precedent_ids:
                        continue 
                    
                    # 2-1. 판례 상세 내용(요지, 판시사항) 가져오기
                    summary_list, holding = get_precedent_detail_text(prec_id)
                    
                    # 2-2. 문서 객체로 변환 및 중복 검사 후 추가
                    full_text, metadata = parse_precedent_content(summary_list, holding, prec_info)
                    
                    if full_text:
                        doc = Document(page_content=full_text, metadata=metadata)
                        all_docs.append(doc)
                        precedent_ids.add(prec_id) 

                page += 1
                
        if not all_docs:
            print("❌ 저장할 판례 데이터가 없어 DB 생성을 건너뜁니다.")
            return

        # 3. 벡터 DB 생성 및 로컬 저장
        print(f"⚡ 총 {len(all_docs)}개 판례 벡터화 및 DB 저장 시작...")
        start_time = time.time()
        
        # Document 객체 리스트를 FAISS 벡터 DB로 변환 및 저장
        self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.vectorstore.save_local(DB_PATH)
        
        elapsed_time = time.time() - start_time
        print(f"✅ 판례 DB 신규 구축 및 저장 완료! (소요시간: {elapsed_time:.1f}초, 경로: {os.path.abspath(DB_PATH)})")
        
    def search_relevant_precedents(self, query, k=2):
        """
        로컬에 로드된 DB에서 사용자 질문과 관련된 판례를 검색합니다.
        
        :param query: 검색을 위한 사용자 질문(텍스트)
        :param k: 반환할 검색 결과(Document)의 최대 개수입니다. (기본값: 2)
        :return: 검색된 판례 내용(page_content) 리스트
        """
        # DB 로드 확인 및 시도
        if not self.vectorstore:
            self.initialize_database()
        
        if not self.vectorstore:
            print("⚠️ 판례 DB가 존재하지 않아 검색을 수행할 수 없습니다.")
            return []
        
        print(f"🔍 판례 DB에서 '{query[:20]}...' 관련 판례 {k}개 검색 중...")
        # 쿼리를 벡터화한 후, DB 내에서 가장 유사한 벡터를 찾아 원문 Document를 반환합니다.
        docs = self.vectorstore.similarity_search(query, k=k) 
        
        # Document의 텍스트 내용만 추출하여 반환
        return [doc.page_content for doc in docs]

# ==========================================
# 🧪 테스트 코드
# ==========================================
if __name__ == "__main__":
    # DB 저장 경로 생성 (없을 경우 대비)
    if not os.path.exists(os.path.dirname(DB_PATH)):
        os.makedirs(os.path.dirname(DB_PATH))

    manager = PrecedentContextManager()
    
    # DB 초기화 (로드 또는 구축)
    manager.initialize_database()
    
    # 구축된 DB로 검색 수행
    question = "직원이 업무 태만으로 해고되었을 때 부당 해고로 인정될 수 있는 기준이 뭐야?"
    relevant_cases = manager.search_relevant_precedents(question, k=1)
    
    print("\n" + "="*50)
    print("📝 검색된 유사 판례:")
    print("="*50)
    
    if relevant_cases:
        print(relevant_cases[0])
    else:
        print("검색 결과가 없습니다.")