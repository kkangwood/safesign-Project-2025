# legal_searchTest.py (수정 완료)

import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import os # os 모듈 import 추가 (FileNotFoundError 확인 시 유용)

# ==============================================================================
# 1. 설정 및 파일 경로 (legal_moel.py에서 저장된 파일 경로와 일치하도록 수정)
# ==============================================================================
KEYWORD = 'moel' # <-- 'moel_retirement'에서 'moel'로 수정
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2' 
INDEX_FILE = f'faiss_index_{KEYWORD}.bin'
METADATA_FILE = f'faiss_metadata_{KEYWORD}.json'
TOP_K = 3 # 가장 유사한 상위 K개의 문서를 검색

# ==============================================================================
# 2. RAG 검색 로직
# ==============================================================================

def load_db():
    """FAISS 인덱스와 메타데이터를 로드합니다."""
    try:
        # 파일 존재 여부 확인 로직 추가 (오류 메시지 명확화)
        if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
             raise FileNotFoundError(f"FAISS 파일({INDEX_FILE} 또는 {METADATA_FILE})을 찾을 수 없습니다.")

        # FAISS 인덱스 로드
        index = faiss.read_index(INDEX_FILE)
        
        # 메타데이터 로드
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata_map = json.load(f)
            
        # 임베딩 모델 로드
        model = SentenceTransformer(MODEL_NAME)
        
        print(f"✅ DB 로드 완료. 벡터 수: {index.ntotal}, 차원: {index.d}")
        return model, index, metadata_map
        
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print("DB 구축 시 사용한 KEYWORD가 현재 설정과 일치하는지 확인해 주세요.")
        return None, None, None
    except Exception as e:
        print(f"❌ DB 로드 중 알 수 없는 오류 발생: {e}")
        return None, None, None


def rag_search(user_query: str, model, index, metadata_map):
    """
    사용자 쿼리를 벡터화하고 FAISS를 검색하여 관련 문서를 반환합니다.
    """
    print(f"\n[검색 시작] 쿼리: '{user_query}'")
    
    # 1. 쿼리 벡터화
    query_vector = model.encode([user_query], convert_to_tensor=True).cpu().numpy().astype('float32')
    
    # 2. FAISS 검색
    # D: 거리(Distance), I: 인덱스(Index)
    D, I = index.search(query_vector, TOP_K) 
    
    print(f"[검색 완료] 가장 유사한 {TOP_K}개 문서를 찾았습니다.")
    
    # 3. 검색 결과 매핑 및 출력
    retrieved_results = []
    
    for rank, (doc_index, distance) in enumerate(zip(I[0], D[0])):
        if doc_index == -1: # 유효하지 않은 인덱스인 경우 건너뜀
            continue
            
        # 메타데이터 매핑
        doc_metadata = metadata_map[doc_index]
        
        # 결과 저장
        score = 1 / (1 + distance)
        retrieved_results.append({
            "rank": rank + 1,
            "score": score,
            "title": doc_metadata['title'],
            "source_id": doc_metadata['id'],
            "detail_url": doc_metadata['detail_url'],
            "content_raw": doc_metadata['chunk_text'] # LLM에게 전달할 핵심 텍스트
        })
        
        # 출력
        print(f"\n--- [검색 결과 {rank + 1}위] ---")
        print(f"제목: {doc_metadata['title']}")
        print(f"유사도 점수: {score:.4f}")
        print(f"원문 ID: {doc_metadata['id']}")
        print(f"원문 내용 (일부):\n{doc_metadata['chunk_text']}")

    return retrieved_results


if __name__ == "__main__":
    # 1. DB 로드
    model, index, metadata_map = load_db()
    
    if model and index and metadata_map:
        # 2. 사용자 쿼리 정의 및 검색 실행
        test_query = "정년이 연장된 경우 조합원 자격은 어떻게 되나요?"
        
        rag_search(test_query, model, index, metadata_map)