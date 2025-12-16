import React, { useState, useEffect, useRef } from 'react';
import { Upload, FileText, Shield, AlertTriangle, ChevronDown, ChevronUp, Key, Info, BookOpen, X } from 'lucide-react';

// ==================================================================================
// [1] 설정 및 API 요청 함수 (Service Layer)
// ==================================================================================

const API_BASE_URL = "http://localhost:8000"; // FastAPI 서버 주소

const apiService = {
  /**
   * 1단계: PDF 업로드 및 텍스트 추출
   */
  uploadPDF: async (file, apiKey) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('api_key', apiKey);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || '파일 업로드 실패');
      }
      return await response.json();
    } catch (error) {
      console.error("Upload Error:", error);
      throw error;
    }
  },

  /**
   * 2단계: AI 분석 요청 (스트리밍)
   */
  analyzeTextStream: async (text, apiKey, onProgress) => {
    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, api_key: apiKey }),
      });

      if (!response.ok) {
         const errData = await response.json();
         throw new Error(errData.detail || '분석 요청 실패');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); 

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            
            if (data.status === 'progress') {
              onProgress(data.current, data.total, data.message);
            } else if (data.status === 'complete') {
              return data.results;
            } else if (data.status === 'error') {
              throw new Error(data.message);
            }
          } catch (e) {
            console.error("Parsing Error:", e);
          }
        }
      }
    } catch (error) {
      console.error("Stream Error:", error);
      throw error;
    }
  }
};


// ==================================================================================
// [2] 메인 컴포넌트 (UI Layer)
// ==================================================================================

function App() {
  // --- 상태 변수 ---
  const [apiKey, setApiKey] = useState('');           
  const [pdfFile, setPdfFile] = useState(null);       
  const [pdfText, setPdfText] = useState('');         
  const [resultList, setResultList] = useState([]);   

  // UI 상태
  const [step, setStep] = useState('upload'); // 'upload' | 'review' | 'result'
  const [isLoading, setIsLoading] = useState(false);
  const [showToxicOnly, setShowToxicOnly] = useState(false);
  const [expandedId, setExpandedId] = useState(null);
  
  // 모달 상태 (새로 추가됨)
  const [modalData, setModalData] = useState(null); // null이면 닫힘, 데이터가 있으면 열림

  // 리사이징 상태
  const [sidebarWidth, setSidebarWidth] = useState(500); 
  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef(null);

  // 진행률 상태 (current/total)
  const [progressStatus, setProgressStatus] = useState({ current: 0, total: 0, message: '' });


  // --- 이벤트 핸들러 ---

  // 1. 파일 선택
  const handleFileUpload = async (e) => {
    const file = e.target.files ? e.target.files[0] : null;
    if (!file) return;
    processUpload(file);
  };

  // 1-1. 업로드 로직
  const processUpload = async (file) => {
    if (!apiKey.trim()) {
      alert('⚠️ Gemini API Key를 먼저 입력해주세요!');
      return;
    }

    setPdfFile(file); 
    setIsLoading(true);
    setProgressStatus({ message: '파일 업로드 및 텍스트 추출 중...' });

    try {
      console.log("파일 전송 중:", file.name);
      const data = await apiService.uploadPDF(file, apiKey);
      setPdfText(data.text);
      setStep('review');
    } catch (error) {
      alert(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // 2. 분석 요청
  const handleAnalyze = async () => {
    setIsLoading(true);
    setProgressStatus({ current: 0, total: 0, message: '분석 준비 중...' });

    try {
      const results = await apiService.analyzeTextStream(
        pdfText, 
        apiKey, 
        (current, total, msg) => {
          setProgressStatus({ current, total, message: msg });
        }
      );

      setResultList(results);
      setStep('result');
      
    } catch (error) {
      alert('분석 중 오류가 발생했습니다: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // 3. 인터랙션 (카드 클릭 -> 스크롤)
  const toggleExpand = (item) => {
    // [수정됨] is_toxic이 True인 항목만 클릭 가능
    if (!item.is_toxic) return;

    setExpandedId(expandedId === item.id ? null : item.id);

    const element = document.getElementById(`line-${item.id}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      element.classList.add('ring-2', 'ring-blue-500');
      setTimeout(() => element.classList.remove('ring-2', 'ring-blue-500'), 1500);
    }
  };

  // 4. 모달 핸들러 (새로 추가됨)
  const openModal = (e, item) => {
    e.stopPropagation(); // 카드 토글 방지
    setModalData(item);
  };

  const closeModal = () => {
    setModalData(null);
  };

  // 5. 리사이징
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

  // 필터링
  const filteredResults = showToxicOnly 
    ? resultList.filter(r => r.is_toxic) 
    : resultList;
  
  const toxicCount = resultList.filter(r => r.is_toxic).length;

  // [수정됨] 리스크 점수 기반 색상 결정 헬퍼 함수
  const getRiskColor = (score) => {
    // 7.0 이상: 빨강 (위험)
    if (score >= 7.0) return {
      bg: "bg-red-50",
      border: "border-red-200",
      badge: "bg-red-100 text-red-700 border-red-200",
      text: "text-red-900"
    };
    // 4.0 이상 7.0 미만: 노랑 (주의)
    if (score >= 4.0) return {
      bg: "bg-yellow-50",
      border: "border-yellow-200",
      badge: "bg-yellow-100 text-yellow-800 border-yellow-200",
      text: "text-yellow-900"
    };
    // 4.0 미만: 초록 (안전)
    return {
      bg: "bg-green-50",
      border: "border-green-200",
      badge: "bg-green-100 text-green-700 border-green-200",
      text: "text-green-900"
    };
  };


  // --- 렌더링 ---
  return (
    <div className="flex h-screen bg-gray-50 font-sans overflow-hidden select-none relative">
      
      {/* 1. 모달 (새로 추가됨) */}
      {modalData && (
        <div className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col overflow-hidden animate-in fade-in zoom-in duration-200">
            {/* 모달 헤더 */}
            <div className="p-5 border-b border-slate-200 flex justify-between items-center bg-slate-50">
              <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-blue-600"/>
                법적 근거 및 판례
              </h3>
              <button onClick={closeModal} className="p-1 rounded-full hover:bg-slate-200 transition-colors">
                <X className="w-6 h-6 text-slate-500" />
              </button>
            </div>
            
            {/* 모달 내용 */}
            <div className="p-6 overflow-y-auto space-y-6">
              <div>
                <h4 className="text-sm font-bold text-slate-500 mb-2 uppercase tracking-wide">판단 근거 조항 (Clause)</h4>
                <div className="p-4 bg-slate-100 rounded-lg text-slate-800 leading-relaxed font-medium border border-slate-200">
                  {modalData.reason}
                </div>
              </div>

              <div>
                <h4 className="text-sm font-bold text-slate-500 mb-2 uppercase tracking-wide">참고 판례 / 법령 (Context Used)</h4>
                <div className="p-4 bg-blue-50 rounded-lg text-slate-800 leading-relaxed text-sm border border-blue-100 whitespace-pre-wrap">
                  {modalData.context_used || "관련된 구체적인 판례나 법령 데이터가 제공되지 않았습니다."}
                </div>
              </div>
            </div>

            {/* 모달 하단 */}
            <div className="p-4 border-t border-slate-200 bg-slate-50 text-right">
              <button onClick={closeModal} className="px-5 py-2 bg-slate-800 hover:bg-slate-900 text-white rounded-lg text-sm font-medium transition-colors">
                닫기
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 2. 사이드바 */}
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
        
        <div className="mt-auto">
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
            <h3 className="flex items-center gap-2 text-sm font-semibold mb-3 text-slate-300">
              <Info className="w-4 h-4" /> 사용 가이드
            </h3>
            <ul className="text-xs text-slate-400 space-y-2 list-disc pl-4">
              <li>PDF 계약서를 업로드하세요.</li>
              <li>자동으로 텍스트가 추출됩니다.</li>
              <li>'분석 시작'을 누르면 AI가 독소 조항을 찾아냅니다.</li>
            </ul>
          </div>
          <p className="text-center text-[10px] text-slate-600 mt-4">Powered by Google Gemini</p>
        </div>
      </aside>

      {/* 3. 메인 영역 */}
      <main className="flex-1 flex flex-col p-8 overflow-hidden relative min-w-[400px]">
        {/* [수정됨] 로딩 오버레이: 프로그레스 바 제거, 뱅글뱅글 도는 스피너 + 메시지 */}
        {isLoading && (
          <div className="absolute inset-0 bg-white/90 backdrop-blur-sm z-50 flex flex-col items-center justify-center p-8">
            <div className="relative">
              <div className="w-16 h-16 border-4 border-slate-200 rounded-full"></div>
              <div className="w-16 h-16 border-4 border-blue-600 rounded-full border-t-transparent animate-spin absolute top-0 left-0"></div>
            </div>
            
            <h3 className="mt-8 text-lg font-bold text-slate-800 animate-pulse">
              {progressStatus.message || "처리 중..."}
            </h3>
            <p className="text-slate-500 text-sm mt-2">잠시만 기다려주세요</p>
          </div>
        )}

        <header className="mb-6">
          <h2 className="text-2xl font-bold text-slate-800">계약서 업로드 및 확인</h2>
        </header>

        <div className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden flex flex-col">
          {step === 'upload' && (
            <div className="flex-1 flex flex-col items-center justify-center m-4">
               <input id="file-upload" type="file" accept=".pdf" className="hidden" onChange={handleFileUpload} />
              <label htmlFor="file-upload" className="flex flex-col items-center justify-center w-full h-full border-2 border-dashed border-slate-300 rounded-xl hover:bg-blue-50 hover:border-blue-400 transition-all cursor-pointer group">
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
                    
                    const matchedResult = resultList.find(r => 
                      line.trim().startsWith(r.clause.substring(0, 15).trim()) || 
                      (r.clause.includes(line.trim()) && line.trim().length > 10)
                    );
                    
                    let highlightClass = "";
                    let riskId = "";
                    
                    if (matchedResult) {
                      riskId = `line-${matchedResult.id}`;
                      const score = matchedResult.risk_score || 0;
                      const colors = getRiskColor(score);
                      
                      // 텍스트 뷰 하이라이트는 조금 더 연하게
                      if (score >= 7.0) highlightClass = "bg-red-100/50 text-red-900 border-b-2 border-red-200";
                      else if (score >= 4.0) highlightClass = "bg-yellow-100/50 text-yellow-900 border-b-2 border-yellow-200";
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

      {/* 4. 분석 결과 사이드바 */}
      {step === 'result' && (
        <aside ref={sidebarRef} className="bg-white border-l border-slate-200 flex flex-col shadow-2xl flex-shrink-0 relative" style={{ width: sidebarWidth }}>
          {/* 리사이징 핸들 */}
          <div onMouseDown={() => setIsResizing(true)} className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize hover:bg-blue-400 transition-colors z-40 flex items-center justify-center group">
            <div className="h-8 w-1 bg-slate-300 rounded-full group-hover:bg-white transition-colors"></div>
          </div>

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
            
            <div className="bg-slate-100 p-1 rounded-lg flex text-sm font-medium">
              <button 
                onClick={() => setShowToxicOnly(false)} 
                className={`flex-1 py-1.5 rounded-md transition-all ${!showToxicOnly ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
              >
                전체 보기
              </button>
              <button 
                onClick={() => setShowToxicOnly(true)} 
                className={`flex-1 py-1.5 rounded-md transition-all flex items-center justify-center gap-1 ${showToxicOnly ? 'bg-white text-red-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
              >
                <AlertTriangle className="w-3 h-3" /> 독소 조항만
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-slate-50">
            {filteredResults.map((item) => {
              const isToxic = item.is_toxic; // True / False 여부
              const riskScore = item.risk_score || 0; // 점수
              const isExpanded = expandedId === item.id;
              
              // [수정됨] risk_score에 따른 색상 구분 (7.0 이상 Red / 4.0 이상 Yellow / 그 외 Green)
              const colors = getRiskColor(riskScore);
              
              return (
                <div 
                  key={item.id} 
                  // [수정됨] 독소 조항(True)인 경우만 클릭 이벤트 핸들러 연결
                  onClick={() => toggleExpand(item)} 
                  className={`rounded-xl border p-4 relative transition-all ${colors.bg} ${colors.border} 
                    ${isToxic ? 'cursor-pointer hover:shadow-md' : 'cursor-default opacity-80'}`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center gap-2">
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${colors.badge}`}>
                        Risk: {riskScore}
                      </span>
                    </div>
                    {/* 독소 조항인 경우에만 화살표 표시 */}
                    {isToxic && (isExpanded ? <ChevronUp className="w-4 h-4 text-slate-400"/> : <ChevronDown className="w-4 h-4 text-slate-400"/>)}
                  </div>
                  
                  {/* 조항 내용 (제목) */}
                  <h4 className={`font-bold text-sm mb-1 line-clamp-2 leading-snug ${colors.text}`}>{item.clause}</h4>
                  
                  {/* [수정됨] 독소 조항이며 확장되었을 때의 내용 */}
                  {isToxic && isExpanded && (
                    <div className="mt-3 pt-3 border-t border-black/5 animate-in slide-in-from-top-2 duration-200">
                      
                      {/* 쉬운 해석 (Suggestion) */}
                      <div className="mb-3">
                        <p className="text-xs font-bold text-blue-600 mb-1 flex items-center gap-1">
                          💡 쉬운 해석 (Suggestion)
                        </p>
                        <p className="text-xs text-slate-700 bg-white/60 p-2.5 rounded border border-blue-100 leading-relaxed">
                          {item.suggestion}
                        </p>
                      </div>

                      {/* [수정됨] 우측 하단 버튼 추가 */}
                      <div className="flex justify-end mt-2">
                        <button 
                          onClick={(e) => openModal(e, item)}
                          className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-white text-[10px] px-3 py-1.5 rounded-full font-medium transition-colors shadow-sm"
                        >
                          <BookOpen className="w-3 h-3" />
                          참고 판례/법령 보기
                        </button>
                      </div>
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