import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
# precedent_search에서 판례 검색 및 파싱 관련 함수만 import합니다.
from .precedent_search import search_precedent_list, get_precedent_detail_text, parse_precedent_content 
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로컬 DB 저장 경로 및 대상 검색어 정의
PREC_DB_PATH = "../data/faiss_precedent_db" 
# DB 구축 시 검색할 핵심 키워드 목록
TARGET_QUERIES = ["부당 해고", "위약 예정", "징벌적 손해배상", "불공정 약정", "근로기준법"]

class PrecedentContextManager:
    """
    판례 데이터를 수집, 임베딩하고 FAISS 벡터 데이터베이스를 관리하여
    RAG(Retrieval-Augmented Generation)에 사용할 문맥을 제공하는 클래스입니다.
    """
    def __init__(self):
        self.vectorstore = None  # FAISS 벡터스토어를 저장할 변수
        self.target_queries = TARGET_QUERIES
        # 한국어 임베딩 모델 로드
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

    def initialize_database(self, max_pages=1, display=10):
        """
        로컬 DB를 확인하여 DB를 로드하거나, 없으면 법제처 API를 통해 새로 구축 후 저장합니다.

        :param max_pages: 각 쿼리당 최대 몇 페이지까지 검색할지 지정합니다. (기본값: 1 페이지)
        :param display: 페이지당 검색할 판례의 최대 개수를 지정합니다. (기본값: 10건)
        """
        if self.vectorstore is not None:
            print("💡 판례 DB가 이미 로드되었습니다.")
            return

        # 1. 로컬 DB 파일 존재 확인 및 로드
        if os.path.exists(PREC_DB_PATH) and os.path.isdir(PREC_DB_PATH):
            print(f"✅ [초기화] 기존 판례 DB 로드 중... (경로: {PREC_DB_PATH})")
            try:
                # DB 로드
                self.vectorstore = FAISS.load_local(PREC_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
                print("✅ [초기화] 판례 DB 로드 완료!")
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
        print(f"\n⚡ 총 {len(all_docs)}개 판례 벡터화 및 DB 저장 시작...")
        self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
        
        os.makedirs(os.path.dirname(PREC_DB_PATH), exist_ok=True)
        self.vectorstore.save_local(PREC_DB_PATH)
        
        print(f"✅ 판례 DB 신규 구축 및 저장 완료! (총 {len(all_docs)}개 판례, 경로: {os.path.abspath(PREC_DB_PATH)})")


    def search_relevant_precedents(self, query, k=2):
        """
        로컬에 로드된 DB에서 사용자 질문과 관련된 판례를 검색합니다.
        
        :param query: 검색을 위한 사용자 질문(텍스트)
        :param k: 반환할 검색 결과(Document)의 최대 개수입니다. (기본값: 2)
        :return: 검색된 판례 내용(page_content) 리스트
        """
        # DB가 로드되지 않았으면 로드 시도
        if not self.vectorstore:
            self.initialize_database()
        
        if not self.vectorstore:
            print("⚠️ 판례 DB가 존재하지 않아 검색을 수행할 수 없습니다.")
            return []
        
        print(f"🔍 판례 DB에서 '{query[:20]}...' 관련 판례 {k}개 검색 중...")
        # 유사도 검색 수행
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]