import React, { useState, useEffect, useRef } from 'react';
import { Upload, FileText, Shield, AlertTriangle, ChevronDown, ChevronUp, Key, Info, CheckCircle } from 'lucide-react';

// ==================================================================================
// [1] 설정 및 API 요청 함수 (Service Layer)
// - 백엔드 통신 로직을 여기서 관리합니다.
// ==================================================================================

const API_BASE_URL = "http://localhost:8000"; // FastAPI 서버 주소

const apiService = {
  /**
   * 1단계: PDF 업로드 및 텍스트 추출
   * @param {File} file - 업로드할 PDF 파일
   * @param {string} apiKey - Gemini API Key
   */
  uploadPDF: async (file, apiKey) => {
    // FormData 생성 (파일 전송용)
    const formData = new FormData();
    formData.append('file', file);
    formData.append('api_key', apiKey); // 백엔드 설계에 맞춰 추가

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData, // 헤더에 Content-Type을 설정하지 않습니다 (브라우저가 자동 설정)
      });

      if (!response.ok) throw new Error('파일 업로드 실패');
      return await response.json(); // { status, text, filename } 반환 기대
    } catch (error) {
      console.error("Upload Error:", error);
      throw error;
    }
  },

  /**
   * 2단계: AI 분석 요청
   * @param {string} text - 분석할 계약서 텍스트
   * @param {string} apiKey - Gemini API Key
   */
  analyzeText: async (text, apiKey) => {
    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          api_key: apiKey
        }),
      });

      if (!response.ok) throw new Error('분석 요청 실패');
      return await response.json(); // { status, results: [...] } 반환 기대
    } catch (error) {
      console.error("Analysis Error:", error);
      throw error;
    }
  }
};


// ==================================================================================
// [2] 더미 데이터 (백엔드 서버가 준비 안 됐을 때 테스트용)
// ==================================================================================
const MOCK_DATA = {
  text: `제1조 (목적)\n본 계약은 갑과 을 사이의 거래에 관한 제반 사항을 규정함을 목적으로 한다.\n\n제3조 (계약의 해지)\n갑은 본 계약 기간 중 언제든지 을에게 별도의 통지 없이 본 계약을 해지할 수 있다. 을은 이에 대해 어떠한 이의도 제기할 수 없다.\n\n제7조 (손해배상)\n을의 귀책사유로 인해 갑에게 손해가 발생한 경우, 을은 갑이 청구하는 일체의 손해를 배상하여야 한다.`,
  results: [
    { id: 1, title: '제3조 (계약의 해지)', score: 0.9, reason: '불공정', description: '갑은 언제든지 통지 없이 해지 가능함.', fix: '30일 전 서면 통지 필요.' },
    { id: 2, title: '제7조 (손해배상)', score: 0.6, reason: '모호함', description: '손해배상 범위가 너무 포괄적임.', fix: '통상적인 손해로 제한 필요.' }
  ]
};


// ==================================================================================
// [3] 메인 컴포넌트 (UI Layer)
// ==================================================================================

function App() {
  // --- [핵심 상태 변수 (State Variables)] ---
  // 요청하신 대로 변수를 상단에 모았습니다.
  const [apiKey, setApiKey] = useState('');           // 사용자 API Key
  const [pdfFile, setPdfFile] = useState(null);       // 업로드한 PDF 파일 객체
  const [pdfText, setPdfText] = useState('');         // 추출된 텍스트 (수정 가능)
  const [resultList, setResultList] = useState([]);   // 분석 결과 리스트

  // --- [UI 제어용 상태] ---
  const [step, setStep] = useState('upload'); // 'upload' | 'review' | 'result'
  const [isLoading, setIsLoading] = useState(false);
  const [showToxicOnly, setShowToxicOnly] = useState(false);
  const [expandedId, setExpandedId] = useState(null);
  
  // 리사이징 관련 상태
  const [sidebarWidth, setSidebarWidth] = useState(500); 
  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef(null);


  // --- [이벤트 핸들러: 비즈니스 로직] ---

  // 1. 파일 선택 및 업로드 처리
  const handleFileUpload = async (e) => {
    // input type="file"에서 선택한 파일 가져오기 (드래그앤드롭 대신 클릭 방식 예시)
    // 실제로는 드롭존이나 input 핸들러에서 호출됨
    const file = e.target.files ? e.target.files[0] : null;
    if (!file) return;
    
    processUpload(file);
  };

  // 1-1. 업로드 프로세스 (드래그앤드롭 or 클릭 공통)
  const processUpload = async (file) => {
    if (!apiKey.trim()) {
      alert('⚠️ Gemini API Key를 먼저 입력해주세요!');
      return;
    }

    setPdfFile(file); // 파일 상태 저장
    setIsLoading(true);

    try {
      // [실제 통신] 주석 해제하여 사용
      console.log("파일 전송 중:", file.name);
      const data = await apiService.uploadPDF(file, apiKey);
      setPdfText(data.text);
      setStep('review');
      setIsLoading(false);

      // [테스트용 Mock] (서버 없이 테스트할 때 사용)
      // setTimeout(() => {
      //   setPdfText(MOCK_DATA.text);
        
      // }, 1000);

    } catch (error) {
      alert('업로드 중 오류가 발생했습니다.');
      setIsLoading(false);
    }
  };

  // 2. 분석 요청 처리
  const handleAnalyze = async () => {
    setIsLoading(true);

    try {
      // [실제 통신] 주석 해제하여 사용
      // const data = await apiService.analyzeText(pdfText, apiKey);
      // setResultList(data.results);

      // [테스트용 Mock]
      setTimeout(() => {
        setResultList(MOCK_DATA.results);
        setStep('result');
        setIsLoading(false);
      }, 2000);

    } catch (error) {
      alert('분석 중 오류가 발생했습니다.');
      setIsLoading(false);
    }
  };

  // 3. UI 인터랙션 (카드 클릭 -> 스크롤 이동)
  const toggleExpand = (item) => {
    if (item.score <= 0.4) return;
    setExpandedId(expandedId === item.id ? null : item.id);

    const element = document.getElementById(`line-${item.id}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      element.classList.add('ring-2', 'ring-blue-500');
      setTimeout(() => element.classList.remove('ring-2', 'ring-blue-500'), 1500);
    }
  };

  // 4. 리사이징 로직
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing) return;
      let newWidth = window.innerWidth - e.clientX;
      const maxWidth = window.innerWidth / 2;
      if (newWidth < 350) newWidth = 350;
      if (newWidth > maxWidth) newWidth = maxWidth;
      setSidebarWidth(newWidth);
    };
    const handleMouseUp = () => { setIsResizing(false); document.body.style.cursor = 'default'; };
    if (isResizing) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  // 필터링 결과
  const filteredResults = showToxicOnly 
    ? resultList.filter(r => r.score > 0.4) 
    : resultList;
  const toxicCount = resultList.filter(r => r.score > 0.4).length;


  // ==================================================================================
  // [4] 렌더링 (View Layer)
  // ==================================================================================
  return (
    <div className="flex h-screen bg-gray-50 font-sans overflow-hidden select-none">
      
      {/* --- 사이드바 --- */}
      <aside className="w-72 bg-slate-900 text-white flex flex-col p-6 shadow-xl z-10 flex-shrink-0">
        <div className="flex items-center gap-3 mb-10">
          <Shield className="w-8 h-8 text-blue-400" />
          <h1 className="text-2xl font-bold tracking-tighter">SafeSign</h1>
        </div>
        <div className="mb-8">
          <label className="block text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wide">Gemini API Key</label>
          <div className="relative">
            <Key className="absolute left-3 top-2.5 w-4 h-4 text-slate-500" />
            <input 
              type="password" placeholder="API Key 입력"
              value={apiKey} onChange={(e) => setApiKey(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg py-2 pl-9 pr-3 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            />
          </div>
        </div>
        {/* ...가이드 내용 생략... */}
      </aside>

      {/* --- 메인 영역 --- */}
      <main className="flex-1 flex flex-col p-8 overflow-hidden relative min-w-[400px]">
        {isLoading && (
          <div className="absolute inset-0 bg-white/80 backdrop-blur-sm z-50 flex flex-col items-center justify-center">
            <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
            <p className="text-slate-600 font-medium animate-pulse">처리 중입니다...</p>
          </div>
        )}

        <header className="mb-6">
          <h2 className="text-2xl font-bold text-slate-800">계약서 업로드 및 확인</h2>
        </header>

        <div className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden flex flex-col">
          {step === 'upload' && (
            <div className="flex-1 flex flex-col items-center justify-center m-4">
               {/* 파일 입력 (숨김 처리 후 라벨로 연결) */}
               <input 
                id="file-upload" 
                type="file" 
                accept=".pdf"
                className="hidden"
                onChange={handleFileUpload}
              />
              <label 
                htmlFor="file-upload"
                className="flex flex-col items-center justify-center w-full h-full border-2 border-dashed border-slate-300 rounded-xl hover:bg-blue-50 hover:border-blue-400 transition-all cursor-pointer group"
              >
                <div className="bg-blue-100 p-4 rounded-full mb-4 group-hover:scale-110 transition-transform">
                  <Upload className="w-8 h-8 text-blue-600" />
                </div>
                <p className="text-lg font-semibold text-slate-700">여기를 클릭하여 PDF 업로드</p>
              </label>
            </div>
          )}

          {(step === 'review' || step === 'result') && (
            <div className="flex flex-col h-full">
               <div className="bg-slate-100 px-4 py-2 border-b border-slate-200 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-slate-500" />
                  <span className="text-xs font-bold text-slate-500 uppercase">Text View</span>
                </div>
              </div>

              {step === 'review' ? (
                <textarea 
                  className="flex-1 p-8 resize-none focus:outline-none text-slate-700 leading-8 font-mono text-sm whitespace-pre-wrap"
                  value={pdfText}
                  onChange={(e) => setPdfText(e.target.value)}
                  spellCheck="false"
                />
              ) : (
                <div className="flex-1 p-8 overflow-y-auto text-slate-700 leading-8 font-mono text-sm bg-white">
                  {pdfText.split('\n').map((line, index) => {
                    if (!line.trim()) return <br key={index} />;
                    const matchedResult = resultList.find(r => line.includes(r.title.split(' (')[0]));
                    
                    let highlightClass = "";
                    let riskId = "";
                    if (matchedResult) {
                      riskId = `line-${matchedResult.id}`;
                      if (matchedResult.score > 0.8) highlightClass = "bg-red-100/80 text-red-900 border-b-2 border-red-200";
                      else if (matchedResult.score > 0.4) highlightClass = "bg-yellow-100/80 text-yellow-900 border-b-2 border-yellow-200";
                    }
                    return <p key={index} id={riskId} className={`mb-2 px-1 rounded transition-colors ${highlightClass}`}>{line}</p>;
                  })}
                </div>
              )}

              {step === 'review' && (
                <div className="p-4 border-t border-slate-100 bg-white text-right">
                  <button onClick={handleAnalyze} className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-bold shadow-lg flex items-center gap-2 ml-auto">
                    <Shield className="w-5 h-5" /> AI 정밀 분석 시작
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* --- 분석 결과 영역 --- */}
      {step === 'result' && (
        <aside ref={sidebarRef} className="bg-white border-l border-slate-200 flex flex-col shadow-2xl flex-shrink-0 relative" style={{ width: sidebarWidth }}>
          <div onMouseDown={() => setIsResizing(true)} className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize hover:bg-blue-400 transition-colors z-50" />
          
          <div className="p-6 border-b border-slate-100">
            <h3 className="text-lg font-bold text-slate-800 mb-4">분석 리포트</h3>
            <div className="flex gap-2 mb-4">
              <div className="flex-1 bg-red-50 border border-red-100 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-red-600">{toxicCount}</div>
                <div className="text-xs text-red-400 font-medium">독소 조항</div>
              </div>
              <div className="flex-1 bg-slate-50 border border-slate-100 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-slate-700">{resultList.length}</div>
                <div className="text-xs text-slate-400 font-medium">전체 조항</div>
              </div>
            </div>
            {/* 필터 버튼 생략 (이전 코드와 동일) */}
             <div className="bg-slate-100 p-1 rounded-lg flex text-sm font-medium">
              <button onClick={() => setShowToxicOnly(false)} className={`flex-1 py-1.5 rounded-md ${!showToxicOnly ? 'bg-white shadow-sm' : 'text-slate-500'}`}>전체</button>
              <button onClick={() => setShowToxicOnly(true)} className={`flex-1 py-1.5 rounded-md ${showToxicOnly ? 'bg-white text-red-600 shadow-sm' : 'text-slate-500'}`}>독소 조항</button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-slate-50">
            {filteredResults.map((item) => {
              const isToxic = item.score > 0.4;
              const isExpanded = expandedId === item.id;
              let cardClass = item.score > 0.8 ? "border-red-200 bg-red-50" : item.score > 0.4 ? "border-yellow-200 bg-yellow-50" : "border-green-200 bg-green-50/30";
              
              return (
                <div key={item.id} onClick={() => toggleExpand(item)} className={`rounded-xl border p-4 relative cursor-pointer ${cardClass}`}>
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-white/50">{item.score > 0.8 ? '고위험' : item.score > 0.4 ? '주의' : '안전'}</span>
                    {isToxic && (isExpanded ? <ChevronUp className="w-4 h-4"/> : <ChevronDown className="w-4 h-4"/>)}
                  </div>
                  <h4 className="font-bold text-slate-800 text-sm mb-1">{item.title}</h4>
                  {isToxic && isExpanded && (
                    <div className="mt-3 space-y-3 border-t border-black/5 pt-3">
                      <p className="text-xs text-slate-700 bg-white/50 p-2 rounded">⚠️ {item.description}</p>
                      <p className="text-xs text-blue-800 bg-blue-50 p-2 rounded">💡 {item.fix}</p>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </aside>
      )}
    </div>
  );
}

export default App;