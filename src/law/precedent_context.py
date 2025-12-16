import os
import time
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Deprecation 경고를 피하기 위해 수정
from langchain_core.documents import Document

# --- 설정 ---
# ⭐️ DB_PATH를 판례 전용으로 변경
DB_PATH = "../data/faiss_precedent_db" 
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli" # 사용할 임베딩 모델 이름
# ⭐️ 판례 데이터셋 ID
DATASET_ID = "joonhok-exo-ai/korean_law_open_data_precedents" 
SAMPLE_SIZE = 1000 # 테스트/구축용 데이터 개수 (전체 사용 시 None)

class PrecedentContextManager:
    """
    판례 데이터셋을 기반으로 벡터 DB를 구축하고 관리하는 클래스입니다.
    """
    def __init__(self):
        self.vectorstore = None
        # 임베딩 모델 객체는 한 번만 생성
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # ⚠️ 참고: self.embeddings 객체를 생성할 때 네트워크 연결이 필요할 수 있습니다.

    def create_database(self):
        """
        Hugging Face 데이터셋에서 판례를 다운로드하고 Document 객체로 변환합니다.
        """
        print(f"📥 판례 데이터셋 다운로드 중... ({DATASET_ID})")
        
        try:
            dataset = load_dataset(DATASET_ID, split="train") 
            
            if SAMPLE_SIZE and len(dataset) > SAMPLE_SIZE:
                dataset = dataset.select(range(SAMPLE_SIZE)) 
                print(f"    - (설정) 상위 {SAMPLE_SIZE}개만 벡터화합니다.")
                
        except Exception as e:
            print(f"❌ 데이터셋 로드 실패: {e}")
            return []
        
        print("🔄 문서 객체(Document)로 변환 중...")
        documents = []

        for item in dataset:
            # 데이터셋 컬럼 매핑
            content = item.get('전문', '')
            summary = item.get('판결요지', '')
            case_name = item.get('사건명', '사건명 정보 없음')
            case_number = item.get('사건번호', 'N/A')

            # 검색 정확도를 위한 page_content 구성
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
            
            if len(summary) > 10: # 요지가 짧은 데이터는 제외
                 documents.append(Document(page_content=page_content, metadata=metadata))
        
        print(f"    - 변환된 유효 문서: {len(documents)}개")
        return documents

    def initialize_database(self):
        """
        로컬 DB 경로를 확인하여 DB를 로드하거나 새로 구축 후 저장합니다.
        """
        if self.vectorstore is not None:
            print("💡 판례 DB가 이미 로드되었습니다.")
            return

        # 1. 로컬 DB 파일 존재 확인 및 로드
        if os.path.exists(DB_PATH) and os.path.isdir(DB_PATH):
            print(f"✅ [초기화] 기존 판례 DB 로드 중... (경로: {DB_PATH})")
            try:
                # 로컬 DB 로드 (allow_dangerous_deserialization=True 설정)
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
        print("📚 [초기화] 판례 데이터 신규 구축을 시작합니다...")
        all_docs = self.create_database()

        if not all_docs:
            print("❌ 저장할 판례 데이터가 없어 DB 생성을 건너뜁니다.")
            return

        # 3. 벡터 DB 생성 및 로컬 저장
        print(f"⚡ 총 {len(all_docs)}개 판례 벡터화 및 DB 저장 시작...")
        start_time = time.time()
        
        self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
        
        # 로컬 저장
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
        # DB가 로드되지 않았으면 로드 시도
        if not self.vectorstore:
            self.initialize_database()
        
        if not self.vectorstore:
            print("⚠️ 판례 DB가 존재하지 않아 검색을 수행할 수 없습니다.")
            return []
        
        print(f"🔍 판례 DB에서 '{query[:20]}...' 관련 판례 {k}개 검색 중...")
        # 유사도 검색 수행
        docs = self.vectorstore.similarity_search(query, k=k)
        
        # 
        
        return [doc.page_content for doc in docs]

# ==========================================
# 🧪 테스트 코드
# ==========================================
if __name__ == "__main__":
    # DB 저장 경로 생성
    if not os.path.exists(os.path.dirname(DB_PATH)):
        os.makedirs(os.path.dirname(DB_PATH))

    manager = PrecedentContextManager()
    
    # DB가 없으면 구축하고, 있으면 로드합니다.
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