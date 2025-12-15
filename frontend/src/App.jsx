import React, { useState, useEffect, useRef } from 'react';
import { Upload, FileText, Shield, AlertTriangle, ChevronDown, ChevronUp, Key, Info } from 'lucide-react';

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
    // 독소 조항이 아니면(is_toxic false) 클릭 방지하고 싶다면 아래 주석 해제
    // if (!item.is_toxic) return; 

    setExpandedId(expandedId === item.id ? null : item.id);

    const element = document.getElementById(`line-${item.id}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      element.classList.add('ring-2', 'ring-blue-500');
      setTimeout(() => element.classList.remove('ring-2', 'ring-blue-500'), 1500);
    }
  };

  // 4. 리사이징
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


  // --- 렌더링 ---
  return (
    <div className="flex h-screen bg-gray-50 font-sans overflow-hidden select-none">
      
      {/* 1. 사이드바 */}
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
        
        {/* 가이드 복구 */}
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

      {/* 2. 메인 영역 */}
      <main className="flex-1 flex flex-col p-8 overflow-hidden relative min-w-[400px]">
        {/* 로딩 & 프로그레스 오버레이 */}
        {isLoading && (
          <div className="absolute inset-0 bg-white/90 backdrop-blur-sm z-50 flex flex-col items-center justify-center p-8">
            <div className="w-16 h-16 border-4 border-slate-100 border-t-blue-600 rounded-full animate-spin mb-6"></div>
            
            <h3 className="text-xl font-bold text-slate-800 mb-2">{progressStatus.message}</h3>
            
            {/* [수정됨] [ 5 / 20 ] 형태 표시 */}
            {step !== 'upload' &&(<>
            <div className="text-3xl font-mono font-bold text-blue-600 mb-4 tracking-widest">
              [ <span className="text-slate-800">{progressStatus.current}</span> / {progressStatus.total || '-'} ]
            </div>

            <div className="w-full max-w-md h-3 bg-slate-200 rounded-full overflow-hidden relative">
              <div 
                className="h-full bg-blue-600 transition-all duration-300 ease-out relative"
                style={{ width: `${progressStatus.total ? (progressStatus.current / progressStatus.total) * 100 : 0}%` }}
              >
                 <div className="absolute top-0 left-0 bottom-0 right-0 bg-gradient-to-r from-transparent via-white/30 to-transparent w-full -translate-x-full animate-shimmer"></div>
              </div>
            </div>
            </>
          )}
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
                    
                    // [수정됨] 하이라이트 매칭 로직 변경 (title -> clause)
                    // 줄이 조항 텍스트의 앞부분(약 10~20자)을 포함하는지 확인
                    const matchedResult = resultList.find(r => 
                      line.trim().startsWith(r.clause.substring(0, 15).trim()) || 
                      r.clause.includes(line.trim()) && line.trim().length > 10
                    );
                    
                    let highlightClass = "";
                    let riskId = "";
                    if (matchedResult) {
                      riskId = `line-${matchedResult.id}`;
                      // 백엔드 키값 (is_toxic) 사용
                      if (matchedResult.is_toxic) {
                         // 점수가 있다면 점수별 색상, 없다면 기본 독소 색상
                         const score = matchedResult.score || 9.0; 
                         if (score > 8.0) highlightClass = "bg-red-100/80 text-red-900 border-b-2 border-red-200";
                         else highlightClass = "bg-yellow-100/80 text-yellow-900 border-b-2 border-yellow-200";
                      }
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

      {/* 3. 분석 결과 사이드바 */}
      {step === 'result' && (
        <aside ref={sidebarRef} className="bg-white border-l border-slate-200 flex flex-col shadow-2xl flex-shrink-0 relative" style={{ width: sidebarWidth }}>
          {/* 리사이징 핸들 */}
          <div onMouseDown={() => setIsResizing(true)} className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize hover:bg-blue-400 transition-colors z-50 flex items-center justify-center group">
            <div className="h-8 w-1 bg-slate-300 rounded-full group-hover:bg-white transition-colors"></div>
          </div>

          {/* 헤더 복구 */}
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
              // 백엔드 키값 사용: item.clause, item.is_toxic, item.suggestion
              const isToxic = item.is_toxic;
              const isExpanded = expandedId === item.id;
              
              let cardClass = isToxic ? "border-red-200 bg-red-50" : "border-green-200 bg-green-50/30";
              let badgeClass = isToxic ? "bg-red-100 text-red-700 border-red-200" : "bg-green-100 text-green-700 border-green-200";
              let statusText = isToxic ? "독소조항" : "안전";

              return (
                <div key={item.id} onClick={() => toggleExpand(item)} className={`rounded-xl border p-4 relative cursor-pointer hover:shadow-md transition-all ${cardClass}`}>
                  <div className="flex justify-between items-start mb-2">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${badgeClass}`}>{statusText}</span>
                    {isToxic && (isExpanded ? <ChevronUp className="w-4 h-4 text-slate-400"/> : <ChevronDown className="w-4 h-4 text-slate-400"/>)}
                  </div>
                  
                  {/* 조항 내용 (제목) */}
                  <h4 className="font-bold text-slate-800 text-sm mb-1 line-clamp-2 leading-snug">{item.clause}</h4>
                  
                  {isToxic && isExpanded && (
                    <div className="mt-3 space-y-3 border-t border-black/5 pt-3">
                      <div>
                        <p className="text-xs font-bold text-slate-500 mb-1">⚠️ 판단 근거</p>
                        <p className="text-xs text-slate-700 bg-white/50 p-2 rounded leading-relaxed">{item.reason}</p>
                      </div>
                      <div>
                        <p className="text-xs font-bold text-blue-600 mb-1">💡 수정 제안</p>
                        <p className="text-xs text-blue-800 bg-blue-50 p-2 rounded leading-relaxed">{item.suggestion}</p>
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