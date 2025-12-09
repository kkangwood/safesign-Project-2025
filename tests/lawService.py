import requests

# 요청 URL
url = 'http://www.law.go.kr/DRF/lawService.do'

params = {
    "OC": "junhajs",                # 사용자 이메일 ID (예: g4c@korea.kr → OC=g4c)
    "target": "moelCgmExpc",     # 서비스 대상
    "type": "JSON",              # 출력 형태 (XML, HTML, JSON 중 선택)
    "ID": "11684",             # 법령해석일련번호
}

# GET 요청 보내기
response = requests.get(url, params=params)

# 응답 상태 확인
if response.status_code == 200:
    print(response.text)
    # data = response.json()  # JSON 파싱
    # print(data)             # 전체 데이터 출력
else:
    print("Error:", response.status_code)