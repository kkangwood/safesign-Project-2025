# rag_moel.py

import requests
import json
import math
import re
from bs4 import BeautifulSoup
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import os

# ==============================================================================
# 1. 환경 설정 및 전역 변수
# ==============================================================================
BASE_URL = 'http://www.law.go.kr'
SEARCH_URL = f'{BASE_URL}/DRF/lawSearch.do'
MAX_ROWS = 20  
DELAY_TIME = 1 
BASE_PARAMS = {
    "OC": "junhajs", 
    "target": "moelCgmExpc",  
    "type": "JSON",           
    "query": "퇴직",           
    "numOfRows": MAX_ROWS,
    "page": 1                 
}

# FAISS 설정
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2' 
KEYWORD = 'moel' # '고용노동부'
INDEX_FILE = f'faiss_index_{KEYWORD}.bin'
METADATA_FILE = f'faiss_metadata_{KEYWORD}.json'


# ==============================================================================
# 2. 데이터 수집 및 클리닝 핵심 함수
# ==============================================================================

def extract_final_detail_url(wrapper_html: str) -> str or None:
    """1차 HTML Wrapper에서 실제 내용 URL 추출"""
    try:
        soup = BeautifulSoup(wrapper_html, 'html.parser')
        url_input = soup.find('input', {'id': 'url'})
        if url_input and url_input.get('value'):
            return url_input.get('value')
        iframe = soup.find('iframe')
        if iframe and iframe.get('src'):
            return iframe.get('src')
        return None
    except Exception:
        return None

def clean_detail_content(detail_html: str) -> str:
    """최종 상세 HTML에서 핵심 해석 텍스트만 추출 및 정제"""
    soup = BeautifulSoup(detail_html, 'html.parser')
    main_content = soup.body
    if not main_content:
        return "콘텐츠 추출 실패"

    text_content = main_content.get_text(separator='\n', strip=True)
    
    start_marker = "【질의요지】"
    if start_marker in text_content:
        text_content = text_content[text_content.find(start_marker):]

    end_marker_1 = "【중앙부처 1차 해석에 대한 안내】"
    end_marker_2 = "검색조문선택"
    end_index = len(text_content)
    
    if end_marker_1 in text_content:
        end_index = min(end_index, text_content.find(end_marker_1))
    if end_marker_2 in text_content:
        end_index = min(end_index, text_content.find(end_marker_2))
        
    text_content = text_content[:end_index].strip()

    final_lines = []
    line_noise_patterns = [
        "본문 바로가기", "고용노동부 누리집", "에서 수집한 데이터입니다.", r"\d{3}-\d{3}-\d{4}" 
    ]

    for line in text_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        is_noise = False
        for pattern in line_noise_patterns:
            if re.search(pattern, line):
                is_noise = True
                break
        if "고용노동부(" in line and len(line) < 50:
             is_noise = True
        if not is_noise:
            final_lines.append(line)

    return '\n'.join(final_lines)

def fetch_and_clean_detail_content(detail_url: str) -> str:
    """1차/2차 요청을 수행하고 클리닝된 텍스트 반환"""
    try:
        wrapper_response = requests.get(detail_url, timeout=30)
        wrapper_response.raise_for_status() 
        wrapper_html = wrapper_response.text
        
        final_url = extract_final_detail_url(wrapper_html)
        if not final_url:
            return "최종 콘텐츠 URL 추출 실패"

        final_response = requests.get(final_url, timeout=30)
        final_response.raise_for_status() 
        final_html = final_response.text
        
        return clean_detail_content(final_html)

    except requests.exceptions.RequestException as e:
        return f"[Error] 요청 실패: {e}"
    except Exception as e:
        return f"[Error] 알 수 없는 오류: {e}"


# ==============================================================================
# 3. 전체 데이터 수집 및 FAISS 저장 로직
# ==============================================================================

def embed_and_save_faiss(data_list):
    """수집된 데이터를 임베딩하고 FAISS 인덱스에 저장"""
    print(f"\n[FAISS] 임베딩 모델 로드 중: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"[ERROR] SentenceTransformer 로드 실패: {e}"); return

    # 텍스트 추출 및 벡터화
    texts = [item['content_raw'] for item in data_list]
    print(f"[FAISS] 총 {len(texts)}개의 텍스트를 벡터화 중...")
    
    embeddings = model.encode(texts, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    D = embeddings_np.shape[1] 

    print(f"[FAISS] 임베딩 완료. 차원: {D}, 벡터 수: {embeddings_np.shape[0]}")

    # FAISS 인덱스 생성 및 추가
    index = faiss.IndexFlatL2(D)
    index.add(embeddings_np)
    
    # 메타데이터 (ID) 매핑 생성
    metadata_map = [
        {
            "id": item['source_id'],
            "title": item['title'],
            "detail_url": item['detail_url'],
            "chunk_text": item['content_raw']
        }
        for item in data_list
    ]

    # 인덱스와 메타데이터 저장
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata_map, f, ensure_ascii=False, indent=4)
    
    print(f"\n[FAISS 저장 성공]")
    print(f" - 인덱스 파일: {os.path.abspath(INDEX_FILE)}")
    print(f" - 메타데이터 파일: {os.path.abspath(METADATA_FILE)}")


def fetch_and_process_all_data():
    """전체 페이지 데이터 수집"""
    all_extracted_data = []
    first_item_output = None
    print(f"[{BASE_PARAMS['query']}] 키워드로 법령 행정해석 데이터 수집 시작...")
    
    initial_params = BASE_PARAMS.copy()
    initial_params['page'] = 1
    
    try:
        initial_response = requests.get(SEARCH_URL, params=initial_params, timeout=30)
        initial_response.raise_for_status()
        initial_data = initial_response.json()
    except Exception as e:
        print(f"초기 API 요청 실패: {e}"); return

    cgmExpc_data = initial_data.get('CgmExpc', {})
    total_count = int(cgmExpc_data.get('totalCnt', 0))
    if total_count == 0:
        print("검색 결과가 없습니다."); return

    total_pages = math.ceil(total_count / MAX_ROWS)
    print(f"총 {total_count}건의 데이터, 총 {total_pages} 페이지 확인.")

    for page in range(1, total_pages + 1):
        print(f"\n--- 페이지 {page}/{total_pages} 데이터 수집 중... ---")
        
        if page == 1:
            list_data = initial_data
        else:
            current_params = BASE_PARAMS.copy()
            current_params['page'] = page
            try:
                list_response = requests.get(SEARCH_URL, params=current_params, timeout=30)
                list_response.raise_for_status()
                list_data = list_response.json()
                time.sleep(DELAY_TIME) 
            except Exception as e:
                print(f"페이지 {page} 요청 실패: {e}"); continue
            
        items = list_data.get('CgmExpc', {}).get('cgmExpc', [])
        for item_index, item in enumerate(items):
            source_id = item.get("법령해석일련번호")
            title = item.get("안건명")
            detail_link = item.get("법령해석상세링크")
            full_detail_url = f"{BASE_URL}{detail_link}"
            
            if page == 1 and item_index == 0:
                 print(f"   - ID {source_id} (첫 항목): 상세 내용 다운로드 시도...")
            
            content_raw = fetch_and_clean_detail_content(full_detail_url)

            extracted = {
                "source_id": source_id, "source_type": "행정해석", "title": title, 
                "issue_date": item.get("해석일자"), "source_agency": item.get("해석기관명"),
                "content_raw": content_raw, "detail_url": full_detail_url
            }
            all_extracted_data.append(extracted)
            
            if page == 1 and item_index == 0:
                first_item_output = extracted
    
    print(f"\n=======================================================")
    print(f"✅ 데이터 수집 완료. 총 {len(all_extracted_data)}건 수집.")
    
    if first_item_output:
        # 첫 번째 항목의 결과 출력 (요청 사항)
        print(f"\n✅ 단일 항목 수집 결과 ({first_item_output['source_id']})")
        print("-------------------------------------------------------")
        for key, value in first_item_output.items():
            if key == 'content_raw':
                print(f"**{key}:**\n{value}") 
            else:
                print(f"**{key}:** {value}")
        print("=======================================================")
        
    if all_extracted_data:
        embed_and_save_faiss(all_extracted_data)


if __name__ == "__main__":
    fetch_and_process_all_data()