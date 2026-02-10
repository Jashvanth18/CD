import { useState } from 'react'
import { 
  LayoutDashboard, 
  BarChart2, 
  Settings, 
  Bell, 
  Search, 
  User,
  ShieldCheck,
  AlertTriangle,
  FileText,
  Briefcase,
  Layers,
  UploadCloud,
  CheckCircle,
  XCircle,
  Camera,
  LogOut,
  Zap,
  Download,
  Filter,
  Eye
} from 'lucide-react'
import jsPDF from 'jspdf'
import autoTable from 'jspdf-autotable'
import './App.css'
import './loader.css'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [selectedFile, setSelectedFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  
  // Real-time stats
  const [stats, setStats] = useState({
    total: 0,
    passed: 0,
    defects: 0,
    alerts: 0
  })

  // Full history
  const [history, setHistory] = useState([])
  const [selectedHistoryItem, setSelectedHistoryItem] = useState(null)

  const handleUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    setSelectedFile(URL.createObjectURL(file))
    setLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      
      // Calculate normalized score (Distance 0-100 -> Confidence 100-0)
      const rawScore = data.similarity || 0;
      const confidence = Math.max(0, Math.min(100, (100 - rawScore)));

      const enrichedData = {
        ...data,
        confidence: confidence,
        timestamp: new Date().toLocaleString(),
        id: Date.now(),
        fileName: file.name
      }

      setResult(enrichedData)
      
      // Update stats
      setStats(prev => ({
        ...prev,
        total: prev.total + 1,
        [data.status === 'OK' ? 'passed' : 'defects']: prev[data.status === 'OK' ? 'passed' : 'defects'] + 1,
        alerts: prev.alerts + (data.status === 'MISSING' ? 1 : 0)
      }))

      // Update history
      setHistory(prev => [enrichedData, ...prev])

    } catch (err) {
      console.error(err)
      alert("Failed to connect to AI Engine")
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status) => {
    switch(status) {
      case 'OK': return '#22C55E';
      case 'MISALIGNED': return '#F59E0B';
      case 'MISSING': return '#EF4444';
      default: return '#666';
    }
  }

  const updateHistoryStatus = (id, newStatus) => {
    setHistory(prevHistory => 
      prevHistory.map(item => 
        item.id === id ? { ...item, status: newStatus } : item
      )
    );
    // Also update current result if it matches
    if (result && result.id === id) {
      setResult(prev => ({ ...prev, status: newStatus }));
    }
    // Update local selection
    if (selectedHistoryItem && selectedHistoryItem.id === id) {
      setSelectedHistoryItem(prev => ({ ...prev, status: newStatus }));
    }
    
    // Update stats logic (simplified approximation)
    setStats(prev => {
       // Re-calculate simply from the new history to be accurate
       // This is expensive but safe for small lists
       const newHist = history.map(item => item.id === id ? { ...item, status: newStatus } : item);
       return {
         total: newHist.length,
         passed: newHist.filter(i => i.status === 'OK').length,
         defects: newHist.filter(i => i.status === 'MISALIGNED').length,
         alerts: newHist.filter(i => i.status === 'MISSING').length
       };
    });
  }

  const getCorrectiveAction = (status, stage) => {
     if (status === 'OK') return "None. Proceed to next stage.";
     if (status === 'MISSING') return "Check component supply. Install missing part immediately.";
     if (status === 'MISALIGNED') return "Stop line. Re-align component to match reference markers.";
     return "Review manual inspection.";
  }

  const downloadReport = () => {
    try {
      const doc = new jsPDF();
      
      // Sort history by stage name for sequenced reporting
      const sortedHistory = [...history].sort((a, b) => a.class_name.localeCompare(b.class_name));

      // Title Page
      doc.setFontSize(22);
      doc.setTextColor(33, 33, 33);
      doc.text("Cloud Destinations", 105, 40, null, null, "center");
      
      doc.setFontSize(16);
      doc.text("Automated Defect Analysis Report", 105, 50, null, null, "center");
      
      doc.setFontSize(12);
      doc.setTextColor(100);
      doc.text(`DATE GENERATED: ${new Date().toLocaleString()}`, 105, 60, null, null, "center");
      
      doc.setLineWidth(0.5);
      doc.line(20, 70, 190, 70);

      // Summary Table
      doc.setTextColor(0);
      doc.setFontSize(14);
      doc.text("Session Summary", 20, 85);

      const summaryData = [
         ["Total Scanned", stats.total],
         ["Passed (OK)", stats.passed],
         ["Defects Found", stats.defects],
         ["Missing Components", stats.alerts]
      ];
      
      autoTable(doc, {
        startY: 90,
        head: [['Metric', 'Count']],
        body: summaryData,
        theme: 'grid',
        headStyles: { fillColor: [99, 102, 241] }, // Indigo color
      });

      // Detailed Individual Reports
      sortedHistory.forEach((item, index) => {
         doc.addPage();
         
         // Header
         doc.setFillColor(245, 247, 250);
         doc.rect(0, 0, 210, 30, 'F');
         doc.setFontSize(16);
         doc.setTextColor(33);
         doc.text(`Inspection Sequence #${index + 1}`, 20, 20);
         
         // Image
         if (item.image) {
            try {
               // Keep aspect ratio roughly 4:3
               const imgProps = doc.getImageProperties(`data:image/jpeg;base64,${item.image}`);
               const pdfWidth = 120;
               const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
               
               doc.addImage(`data:image/jpeg;base64,${item.image}`, 'JPEG', 20, 40, pdfWidth, pdfHeight);

               // Draw a border around the image
               doc.setDrawColor(200);
               doc.rect(20, 40, pdfWidth, pdfHeight);
               
               // Details Box next to image
               const startX = 150;
               const startY = 40;
               
               doc.setFontSize(12);
               doc.setFont("helvetica", "bold");
               doc.setTextColor(0);
               doc.text("Result Status:", startX, startY);
               
               // Color code the status text
               if (item.status === 'OK') doc.setTextColor(34, 197, 94); // Green
               else if (item.status === 'MISSING') doc.setTextColor(239, 68, 68); // Red
               else doc.setTextColor(245, 158, 11); // Orange
               
               doc.text(item.status, startX, startY + 7);
               
               doc.setTextColor(0); // Reset black
               doc.setFont("helvetica", "bold");
               doc.text("Confidence:", startX, startY + 20);
               doc.setFont("helvetica", "normal");
               doc.text(`${item.confidence.toFixed(1)}%`, startX, startY + 27);
               
               doc.setFont("helvetica", "bold");
               doc.text("Stage:", startX, startY + 40);
               doc.setFont("helvetica", "normal");
               // Handle long stage names
               const splitStage = doc.splitTextToSize(item.class_name, 50);
               doc.text(splitStage, startX, startY + 47);

               // Corrective Action
               doc.setFont("helvetica", "bold");
               doc.text("Action Required:", startX, startY + 65);
               doc.setFont("helvetica", "normal");
               doc.setFontSize(10);
               const splitAction = doc.splitTextToSize(getCorrectiveAction(item.status, item.class_name), 50);
               doc.text(splitAction, startX, startY + 72);
               doc.setFontSize(12);


               doc.setFont("helvetica", "bold");
               doc.text("File Name:", startX, startY + 90);
               doc.setFont("helvetica", "normal");
               doc.text(String(item.fileName || "unknown").substring(0, 15), startX, startY + 97);

               doc.setFont("helvetica", "bold");
               doc.text("Timestamp:", startX, startY + 110);
               doc.setFont("helvetica", "normal");
               doc.text(item.timestamp, startX, startY + 117);

            } catch (e) {
               console.error("PDF Image Error", e);
               doc.text("Image load error", 20, 50);
            }
         }
      });

      doc.save("CloudDestinations_Inspection_Report.pdf");
      
    } catch (err) {
      console.error(err);
      alert("Error generating PDF: " + err.message);
    }
  }

  // --- SUB-COMPONENTS ---
  
  const HistoryModal = () => {
    if (!selectedHistoryItem) return null;
    return (
      <div style={{
        position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, 
        background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', 
        zIndex: 1000
      }}>
        <div style={{background: 'white', padding: '2rem', borderRadius: '20px', maxWidth: '800px', width: '90%', maxHeight: '90vh', overflowY: 'auto'}}>
          <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '1rem'}}>
            <h2>Inspection Details</h2>
            <div style={{display: 'flex', gap: '10px'}}>
               <button onClick={() => updateHistoryStatus(selectedHistoryItem.id, 'OK')} style={{padding: '5px 10px', background: '#22C55E', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', fontWeight: 'bold'}}>Mark Passed (Override)</button>
               <button onClick={() => updateHistoryStatus(selectedHistoryItem.id, 'MISSING')} style={{padding: '5px 10px', background: '#EF4444', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', fontWeight: 'bold'}}>Mark Missing</button>
               <button onClick={() => setSelectedHistoryItem(null)} style={{background: 'none', border: 'none', cursor: 'pointer'}}><XCircle /></button>
            </div>
          </div>
          
          <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem'}}>
            <img 
              src={`data:image/jpeg;base64,${selectedHistoryItem.image}`} 
              style={{width: '100%', borderRadius: '10px'}} 
              alt="Scan" 
            />
            <div className="checklist">
              <div className="check-item"><span>Status</span><strong style={{color: getStatusColor(selectedHistoryItem.status)}}>{selectedHistoryItem.status}</strong></div>
              <div className="check-item"><span>Time</span><span>{selectedHistoryItem.timestamp}</span></div>
              <div className="check-item"><span>Stage</span><span>{selectedHistoryItem.class_name}</span></div>
              <div className="check-item"><span>Confidence</span><span>{selectedHistoryItem.confidence.toFixed(1)}%</span></div>
              <div className="check-item">
                <span>Keypoint Match Score</span>
                <span title="The number of visual features (ORB) matched between the image and the reference standard. Higher is better (more alignment).">
                  {selectedHistoryItem.orb_metrics?.matches || 0}
                </span>
              </div>
              <div className="check-item"><span>Raw Distance</span><span>{selectedHistoryItem.similarity?.toFixed(2)}</span></div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  const RenderContent = () => {
    switch (activeTab) {
      case 'history':
        return (
          <div className="upload-zone-container" style={{height: '100%'}}>
            <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '20px'}}>
              <h2>Inspection History</h2>
              <button className="tag" onClick={downloadReport} style={{cursor: 'pointer'}}>
                <Download size={16} /> Download Report
              </button>
            </div>
            <div style={{display: 'grid', gap: '10px'}}>
              {history.map(item => (
                 <div key={item.id} className="check-item" style={{alignItems: 'center', padding: '15px', background: '#f9f9f9', borderRadius: '10px', cursor: 'pointer'}} onClick={() => setSelectedHistoryItem(item)}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '15px'}}>
                      <div style={{width: 40, height: 40, background: '#eee', borderRadius: '5px', overflow: 'hidden'}}>
                        <img src={`data:image/jpeg;base64,${item.image}`} style={{width: '100%', height: '100%', objectFit: 'cover'}} />
                      </div>
                      <div>
                        <div style={{fontWeight: 'bold'}}>{item.fileName}</div>
                        <div style={{fontSize: '0.8rem', color: '#666'}}>{item.timestamp}</div>
                      </div>
                    </div>
                    <div className="tag" style={{background: getStatusColor(item.status), color: 'white'}}>
                      {item.status}
                    </div>
                 </div>
              ))}
              {history.length === 0 && <p style={{textAlign: 'center', color: '#999'}}>No history yet.</p>}
            </div>
          </div>
        )

      case 'batch':
        // Group by Stage
        const byStage = history.reduce((acc, item) => {
          const stage = item.class_name || 'Unknown';
          if (!acc[stage]) acc[stage] = [];
          acc[stage].push(item);
          return acc;
        }, {});

        return (
          <div className="upload-zone-container">
            <h2>Batch Analysis (Stage-wise)</h2>
            {Object.keys(byStage).length === 0 ? <p>No data available. Run inspections first.</p> : (
               Object.entries(byStage).map(([stage, items]) => (
                <div key={stage} style={{marginBottom: '2rem'}}>
                  <h3 style={{borderBottom: '2px solid #eee', paddingBottom: '10px'}}>{stage}</h3>
                  <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '1rem'}}>
                    {items.map(item => (
                       <div key={item.id} className="score-card" style={{padding: '1rem', cursor: 'pointer'}} onClick={() => setSelectedHistoryItem(item)}>
                          <div style={{height: '150px', overflow: 'hidden', borderRadius: '10px', marginBottom: '1rem'}}>
                            <img src={`data:image/jpeg;base64,${item.image}`} style={{width: '100%'}} />
                          </div>
                          <div style={{display: 'flex', justifyContent: 'space-between'}}>
                             <strong style={{color: getStatusColor(item.status)}}>{item.status}</strong>
                             <span>{item.confidence.toFixed(1)}%</span>
                          </div>
                       </div>
                    ))}
                  </div>
                </div>
               ))
            )}
          </div>
        )

      case 'reports':
        return (
          <div className="upload-zone-container" style={{display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh'}}>
             <FileText size={64} color="#6366F1" />
             <h2>Generate PDF Report</h2>
             <p>Includes all recent inspections, statistics, and images.</p>
             <button 
               onClick={downloadReport}
               style={{
                 background: '#6366F1', color: 'white', border: 'none', 
                 padding: '1rem 2rem', borderRadius: '50px', fontSize: '1.2rem', 
                 cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '10px', marginTop: '1rem'
               }}
             >
               <Download /> Download Full Report
             </button>
          </div>
        )

      case 'dashboard':
      default:
        return (
          <div className="dashboard-grid">
            <div className="left-column">
              <div className="insights-row">
                <div className="insight-card blue">
                  <div className="card-icon"><Zap size={24} color="#3730A3" /></div>
                  <div>
                    <h3 style={{margin: '0 0 5px 0'}}>Efficiency Update</h3>
                    <p style={{margin: 0, fontSize: '0.9rem', opacity: 0.8}}>System running optimally.</p>
                  </div>
                </div>
                <div className="insight-card green">
                  <div className="card-icon"><CheckCircle size={24} color="#166534" /></div>
                  <div>
                    <h3 style={{margin: '0 0 5px 0'}}>System Calibration</h3>
                    <p style={{margin: 0, fontSize: '0.9rem', opacity: 0.8}}>Sensors aligned.</p>
                  </div>
                </div>
              </div>

              <div className="stats-row">
                <div className="stat-card yellow">
                  <span className="stat-num">{stats.total}</span>
                  <span>Total Scans</span>
                </div>
                <div className="stat-card blue">
                  <span className="stat-num">{stats.passed}</span>
                  <span>Passed OK</span>
                </div>
                <div className="stat-card orange">
                  <span className="stat-num">{stats.defects}</span>
                  <span>Defects</span>
                </div>
                <div className="stat-card green">
                  <span className="stat-num">{stats.alerts}</span>
                  <span>Missing</span>
                </div>
              </div>

              <div className="upload-zone-container">
                <div style={{marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
                  <h3 style={{margin: 0}}>Live Inspection</h3>
                </div>

                <label className="upload-area">
                  <input type="file" className="hidden-input" style={{display:'none'}} onChange={handleUpload} accept="image/*" disabled={loading} />
                  
                  {loading && selectedFile ? (
                    <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%'}}>
                       <div style={{position: 'relative', height: '200px', width: '100%', marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f8f8f8', borderRadius: '10px', overflow: 'hidden'}}>
                          <img src={selectedFile} alt="Scanning..." style={{height: '100%', width: '100%', objectFit: 'contain', opacity: 0.6}} />
                          <div style={{position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)'}}>
                             <div className="loader"></div>
                          </div>
                       </div>
                       <div style={{fontWeight: 600, fontSize: '1.2rem', color: '#6366F1', display: 'flex', alignItems: 'center', gap: '10px'}}>
                          <span>Processing Image...</span>
                       </div>
                    </div>
                  ) : (
                    <>
                      <div style={{marginBottom: '1rem'}}>
                         <UploadCloud size={48} color="#6366F1" />
                      </div>
                      <div style={{fontWeight: 600, fontSize: '1.2rem', color: '#1A1A1A'}}>
                        Click to Upload Inspection Image
                      </div>
                    </>
                  )}
                </label>

                {result && (
                  <div className="result-display">
                    <div className="result-img-wrapper" onClick={() => setSelectedHistoryItem(result)} style={{cursor: 'pointer'}}>
                      <img src={`data:image/jpeg;base64,${result.image}`} className="result-img" alt="Analyzed" />
                    </div>
                    
                    <div className="analysis-tags">
                      <div className="tag">
                        Status: <strong style={{color: getStatusColor(result.status), marginLeft: '5px'}}>{result.status}</strong>
                      </div>
                      <div className="tag">
                         Stage: {result.class_name}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="right-column">
              <div className="score-card">
                <h3>Model Confidence</h3>
                <div className="radial-progress" style={{background: `conic-gradient(${result ? getStatusColor(result.status) : '#3B82F6'} ${result? result.confidence : 0}%, #F3F4F6 0deg)`}}>
                  <div className="radial-inner">
                    {result ? Math.round(result.confidence) : 0}%
                  </div>
                </div>
                <p style={{color: '#666', fontSize: '0.9rem'}}>Confidence Level (Inverse Distance)</p>
                
                <div className="checklist">
                  <h4>Metric Analysis</h4>
                  <div className="check-item">
                    <span>Raw Distance</span>
                    <span style={{color: 'green'}}>{result?.similarity ? (result.similarity).toFixed(2) : '-'}</span>
                  </div>
                  <div className="check-item">
                    <span>Keypoint Match Score</span>
                    <span title="The number of visual features (ORB) matched between the image and the reference standard. Higher is better (more alignment).">
                      {result?.orb_metrics?.matches || '-'}
                    </span>
                  </div>
                  <div className="check-item">
                    <span>Validation</span>
                    <span style={{color: '#3B82F6'}}>{result ? 'Complete' : '-'}</span>
                  </div>
                </div>
              </div>

              <div className="score-card" style={{textAlign: 'left'}}>
                <h3>Recent Activity</h3>
                <div style={{display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem'}}>
                  {history.slice(0, 5).map(log => (
                    <div key={log.id} style={{display: 'flex', gap: '1rem', alignItems: 'center', cursor: 'pointer'}} onClick={() => setSelectedHistoryItem(log)}>
                      <div style={{
                        width: '40px', height: '40px', 
                        background: '#f3f4f6', 
                        borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center'
                      }}>
                        {log.status === 'OK' ? <CheckCircle size={20} color="#22C55E" /> : <AlertTriangle size={20} color="#EF4444" />}
                      </div>
                      <div>
                        <div style={{fontWeight: 600}}>{log.status}</div>
                        <div style={{fontSize: '0.8rem', color: '#666'}}>{log.timestamp.split(',')[1]}</div>
                      </div>
                    </div>
                  ))}
                  {history.length === 0 && <span style={{color: '#999', fontSize: '0.8rem'}}>No recent activity</span>}
                </div>
              </div>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div>
          <div className="brand">
            <ShieldCheck size={32} color="#6366F1" />
            <span>Cloud Destinations</span>
          </div>
          
          <nav className="nav-menu">
            <div className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
              <LayoutDashboard size={20} />
              <span>Dashboard</span>
            </div>
            <div className={`nav-item ${activeTab === 'batch' ? 'active' : ''}`} onClick={() => setActiveTab('batch')}>
              <Layers size={20} />
              <span>Batch Analysis</span>
            </div>
            <div className={`nav-item ${activeTab === 'reports' ? 'active' : ''}`} onClick={() => setActiveTab('reports')}>
              <FileText size={20} />
              <span>Reports</span>
            </div>
            <div className={`nav-item ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>
              <Briefcase size={20} />
              <span>History</span>
            </div>
          </nav>
        </div>

        <div>
           <div className="nav-item">
            <Settings size={20} />
            <span>Settings</span>
          </div>
          <div className="user-profile">
            <div style={{width: 35, height: 35, borderRadius: '50%', background: '#333', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white'}}>A</div>
            <div style={{flex: 1}}>
              <div style={{color: 'white', fontSize: '0.9rem'}}>Admin</div>
              <div style={{fontSize: '0.8rem'}}>Supervisor</div>
            </div>
            <LogOut size={16} />
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="header">
          <div className="search-bar">
            <Search size={20} />
            <input type="text" placeholder="Search logs..." className="search-input" />
          </div>
          <div className="top-icons">
            <button className="icon-btn"><Bell size={20} /></button>
          </div>
        </header>

        {/* Dynamic Content */}
        <div style={{marginTop: '2rem'}}>
          {RenderContent()}
        </div>

      </main>

      {/* Popup Modal */}
      <HistoryModal />
    </div>
  )
}

export default App
