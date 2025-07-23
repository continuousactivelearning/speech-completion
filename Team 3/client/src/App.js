import React, { useState, useRef, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { Brain, Zap, Target, TrendingUp, Clock, CheckCircle, AlertCircle } from 'lucide-react';
import "./App.css";

function App() {
  const [inputText, setInputText] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Speech-to-text states
  const [isListening, setIsListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const recognitionRef = useRef(null);
  
  // Analytics states
  const [analytics, setAnalytics] = useState({
    wordCount: 0,
    sentenceCount: 0,
    avgWordsPerSentence: 0,
    readingTime: 0,
    complexity: 'Medium'
  });
  
  // History state
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [inputMode, setInputMode] = useState('text'); // 'text' or 'speech'
  
  // New enhanced analytics states
  const [processingSteps, setProcessingSteps] = useState([]);
  const [featureImportance, setFeatureImportance] = useState([]);
  const [noveltyTimeline, setNoveltyTimeline] = useState([]);
  const [topicClusters, setTopicClusters] = useState([]);
  const [confidenceMetrics, setConfidenceMetrics] = useState({});
  const [backendAnalytics, setBackendAnalytics] = useState(null);

  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      setSpeechSupported(true);
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';
      
      recognitionRef.current.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }
        
        setInputText(prev => prev + finalTranscript);
      };
      
      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setError(`Speech recognition error: ${event.error}`);
        setIsListening(false);
      };
      
      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
  }, []);

  // Calculate text analytics
  useEffect(() => {
    if (inputText.trim()) {
      const words = inputText.trim().split(/\s+/);
      const sentences = inputText.split(/[.!?]+/).filter(s => s.trim().length > 0);
      const avgWordsPerSentence = words.length / Math.max(sentences.length, 1);
      const readingTime = Math.ceil(words.length / 200); // Average reading speed
      
      // Simple complexity calculation based on avg words per sentence
      let complexity = 'Simple';
      if (avgWordsPerSentence > 15) complexity = 'Medium';
      if (avgWordsPerSentence > 25) complexity = 'Complex';
      
      setAnalytics({
        wordCount: words.length,
        sentenceCount: sentences.length,
        avgWordsPerSentence: Math.round(avgWordsPerSentence * 10) / 10,
        readingTime,
        complexity
      });
    } else {
      setAnalytics({
        wordCount: 0,
        sentenceCount: 0,
        avgWordsPerSentence: 0,
        readingTime: 0,
        complexity: 'Medium'
      });
    }
  }, [inputText]);

  // Load backend analytics
  useEffect(() => {
    const loadBackendAnalytics = async () => {
      try {
        const response = await fetch("http://localhost:3001/analytics");
        if (response.ok) {
          const data = await response.json();
          setBackendAnalytics(data);
        }
      } catch (error) {
        console.error("Failed to load backend analytics:", error);
      }
    };
    
    loadBackendAnalytics();
    // Refresh every 30 seconds
    const interval = setInterval(loadBackendAnalytics, 30000);
    return () => clearInterval(interval);
  }, []);

  // Generate mock detailed analytics data
  const generateDetailedAnalytics = (text, predictionValue) => {
    const words = text.trim().split(/\s+/);
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    // Feature importance (mock data based on text characteristics)
    const features = [
      { name: 'Word Count', value: Math.min(words.length / 10, 20) },
      { name: 'Sentence Length', value: Math.min(words.length / sentences.length, 25) },
      { name: 'Semantic Coherence', value: Math.random() * 30 + 10 },
      { name: 'Topic Transitions', value: Math.random() * 20 + 5 },
      { name: 'Conclusion Signals', value: Math.random() * 15 + 5 },
      { name: 'Punctuation Density', value: (text.match(/[.!?]/g) || []).length * 2 }
    ];
    
    // Novelty timeline (mock data)
    const noveltyData = sentences.map((_, index) => ({
      sentence: index + 1,
      novelty: Math.random() * 0.8 + 0.2 - (index * 0.05), // Decreasing novelty over time
      trend: index * 2
    }));
    
    // Topic clusters (mock data)
    const clusters = [
      { name: 'Introduction', value: 25, color: '#8884d8' },
      { name: 'Main Points', value: 45, color: '#82ca9d' },
      { name: 'Examples', value: 20, color: '#ffc658' },
      { name: 'Conclusion', value: 10, color: '#ff7300' }
    ];
    
    // Confidence metrics
    const confidence = {
      overall: Math.random() * 20 + 75,
      textStructure: Math.random() * 15 + 80,
      semanticAnalysis: Math.random() * 20 + 70,
      temporalPatterns: Math.random() * 25 + 65
    };
    
    setFeatureImportance(features);
    setNoveltyTimeline(noveltyData);
    setTopicClusters(clusters);
    setConfidenceMetrics(confidence);
  };

  // Simulate processing steps
  const simulateProcessingSteps = async () => {
    const steps = [
      { id: 1, name: 'Text Preprocessing', status: 'pending', icon: <Brain className="w-4 h-4" /> },
      { id: 2, name: 'Feature Extraction', status: 'pending', icon: <Zap className="w-4 h-4" /> },
      { id: 3, name: 'Semantic Embedding', status: 'pending', icon: <Target className="w-4 h-4" /> },
      { id: 4, name: 'Clustering Analysis', status: 'pending', icon: <TrendingUp className="w-4 h-4" /> },
      { id: 5, name: 'Prediction Generation', status: 'pending', icon: <CheckCircle className="w-4 h-4" /> }
    ];
    
    setProcessingSteps(steps);
    
    // Simulate step-by-step processing
    for (let i = 0; i < steps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 600));
      setProcessingSteps(prev => prev.map(step => 
        step.id === i + 1 ? { ...step, status: 'processing' } : step
      ));
      
      await new Promise(resolve => setTimeout(resolve, 400));
      setProcessingSteps(prev => prev.map(step => 
        step.id === i + 1 ? { ...step, status: 'complete' } : step
      ));
    }
  };

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    } else {
      recognitionRef.current?.start();
      setIsListening(true);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!inputText.trim()) {
      setError("Please enter some text first");
      return;
    }
    
    setLoading(true);
    setPrediction(null);
    setError(null);
    setProcessingSteps([]);
    setFeatureImportance([]);
    setNoveltyTimeline([]);
    setTopicClusters([]);
    setConfidenceMetrics({});

    console.log("Sending request with transcript:", inputText);

    // Start processing visualization
    simulateProcessingSteps();

    try {
      const response = await fetch("http://localhost:3001/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ transcript: inputText }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`HTTP ${response.status}: ${errorData.error || 'Unknown error'}`);
      }

      const data = await response.json();
      console.log("Received response:", data);
      setPrediction(data.prediction);
      
      // Generate detailed analytics
      generateDetailedAnalytics(inputText, data.prediction);
      
      // Add to history
      const newEntry = {
        id: Date.now(),
        text: inputText.substring(0, 100) + (inputText.length > 100 ? '...' : ''),
        prediction: data.prediction,
        timestamp: new Date().toLocaleTimeString(),
        wordCount: analytics.wordCount
      };
      
      setPredictionHistory(prev => [newEntry, ...prev.slice(0, 4)]); // Keep last 5
      
    } catch (error) {
      console.error("Error while calling /predict:", error);
      setError(`Error: ${error.message}`);
    }

    setLoading(false);
  };

  const testConnection = async () => {
    try {
      const response = await fetch("http://localhost:3001/test");
      const data = await response.json();
      console.log("Backend connection test:", data);
      setError(null);
      alert("Backend connection successful!");
    } catch (error) {
      console.error("Backend connection failed:", error);
      setError("Backend connection failed");
    }
  };

  const clearText = () => {
    setInputText("");
    setPrediction(null);
    setError(null);
    setProcessingSteps([]);
    setFeatureImportance([]);
    setNoveltyTimeline([]);
    setTopicClusters([]);
    setConfidenceMetrics({});
  };

  const getProgressColor = (percentage) => {
    if (percentage < 30) return '#ff6b6b';
    if (percentage < 70) return '#ffd93d';
    return '#6bcf7f';
  };

  const getComplexityColor = (complexity) => {
    switch (complexity) {
      case 'Simple': return '#6bcf7f';
      case 'Medium': return '#ffd93d';
      case 'Complex': return '#ff6b6b';
      default: return '#6bcf7f';
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>Speech Completion Predictor</h1>
        <p className="subtitle">AI-powered speech progress analysis with real-time insights</p>
      </header>

      <div className="dashboard-grid">
        {/* Main Input Section */}
        <div className="card main-input">
          <div className="card-header">
            <h2>Input</h2>
            <div className="input-mode-toggle">
              <button 
                className={`mode-btn ${inputMode === 'text' ? 'active' : ''}`}
                onClick={() => setInputMode('text')}
              >
                📝 Text
              </button>
              <button 
                className={`mode-btn ${inputMode === 'speech' ? 'active' : ''}`}
                onClick={() => setInputMode('speech')}
                disabled={!speechSupported}
              >
                🎤 Speech
              </button>
            </div>
          </div>

          {inputMode === 'text' && (
            <div className="text-input-section">
              <textarea
                placeholder="Paste your speech text here or switch to speech mode to use voice input..."
                rows={8}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="main-textarea"
              />
            </div>
          )}

          {inputMode === 'speech' && (
            <div className="speech-input-section">
              <div className="speech-controls">
                <button 
                  onClick={toggleListening}
                  className={`speech-btn ${isListening ? 'listening' : ''}`}
                  disabled={!speechSupported}
                >
                  {isListening ? '🔴 Stop Recording' : '🎤 Start Recording'}
                </button>
                {!speechSupported && (
                  <p className="warning">Speech recognition not supported in this browser</p>
                )}
              </div>
              
              <div className="live-transcript">
                <h4>Live Transcript:</h4>
                <div className="transcript-box">
                  {inputText || "Click 'Start Recording' to begin..."}
                </div>
              </div>
            </div>
          )}

          <div className="action-buttons">
            <button onClick={handlePredict} disabled={loading || !inputText.trim()} className="predict-btn">
              {loading ? "🔄 Analyzing..." : "Predict Completion"}
            </button>
            <button onClick={clearText} className="clear-btn">Clear</button>
            <button onClick={testConnection} className="test-btn">Test Backend</button>
          </div>
        </div>

        {/* Processing Pipeline Visualization */}
        {processingSteps.length > 0 && (
          <div className="card processing-pipeline">
            <div className="card-header">
              <h3>🔄 Processing Pipeline</h3>
            </div>
            <div className="processing-steps">
              {processingSteps.map((step) => (
                <div key={step.id} className={`processing-step ${step.status}`}>
                  <div className="step-icon">{step.icon}</div>
                  <div className="step-content">
                    <span className="step-name">{step.name}</span>
                    <div className="step-status">
                      {step.status === 'pending' && <Clock className="w-3 h-3 text-gray-400" />}
                      {step.status === 'processing' && <div className="spinner"></div>}
                      {step.status === 'complete' && <CheckCircle className="w-3 h-3 text-green-500" />}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Analytics Panel */}
        <div className="card analytics-panel">
          <div className="card-header">
            <h3>Text Analytics</h3>
          </div>
          <div className="analytics-grid">
            <div className="metric">
              <span className="metric-value">{analytics.wordCount}</span>
              <span className="metric-label">Words</span>
            </div>
            <div className="metric">
              <span className="metric-value">{analytics.sentenceCount}</span>
              <span className="metric-label">Sentences</span>
            </div>
            <div className="metric">
              <span className="metric-value">{analytics.avgWordsPerSentence}</span>
              <span className="metric-label">Avg Words/Sentence</span>
            </div>
            <div className="metric">
              <span className="metric-value">{analytics.readingTime}m</span>
              <span className="metric-label">Reading Time</span>
            </div>
          </div>
          <div className="complexity-indicator">
            <span>Complexity: </span>
            <span 
              className="complexity-badge" 
              style={{ backgroundColor: getComplexityColor(analytics.complexity) }}
            >
              {analytics.complexity}
            </span>
          </div>
        </div>

        {/* Enhanced Prediction Result */}
        {prediction !== null && (
          <div className="card prediction-result enhanced">
            <div className="card-header">
              <h3>Completion Prediction</h3>
              <div className="confidence-indicator">
                <span className="confidence-label">Confidence:</span>
                <span className="confidence-value">{Math.round(confidenceMetrics.overall || 85)}%</span>
              </div>
            </div>
            <div className="prediction-display">
              <div className="prediction-main">
                <div className="progress-circle-large">
                  <div 
                    className="progress-fill-large" 
                    style={{ 
                      background: `conic-gradient(${getProgressColor(prediction)} ${prediction * 3.6}deg, #e0e0e0 0deg)` 
                    }}
                  >
                    <div className="progress-inner-large">
                      <span className="progress-value-large">{prediction}%</span>
                      <span className="progress-label-large">Complete</span>
                    </div>
                  </div>
                </div>
                <div className="prediction-insights">
                  <h4>Analysis Results</h4>
                  <p>
                    {prediction < 30 && "🟡 Speech appears to be in early stages. Consider developing main points further."}
                    {prediction >= 30 && prediction < 70 && "🟠 Speech is progressing well. You're building toward a strong conclusion."}
                    {prediction >= 70 && "🟢 Speech appears to be nearing completion. Consider summarizing key points."}
                  </p>
                  <div className="confidence-breakdown">
                    <div className="confidence-item">
                      <span>Text Structure:</span>
                      <span>{Math.round(confidenceMetrics.textStructure || 80)}%</span>
                    </div>
                    <div className="confidence-item">
                      <span>Semantic Analysis:</span>
                      <span>{Math.round(confidenceMetrics.semanticAnalysis || 75)}%</span>
                    </div>
                    <div className="confidence-item">
                      <span>Temporal Patterns:</span>
                      <span>{Math.round(confidenceMetrics.temporalPatterns || 70)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Feature Importance Chart */}
        {featureImportance.length > 0 && (
          <div className="card feature-importance">
            <div className="card-header">
              <h3>Feature Importance</h3>
            </div>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={featureImportance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#667eea" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Novelty Timeline */}
        {noveltyTimeline.length > 0 && (
          <div className="card novelty-timeline">
            <div className="card-header">
              <h3>Semantic Novelty Timeline</h3>
            </div>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={noveltyTimeline}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="sentence" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="novelty" stroke="#82ca9d" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Topic Clusters */}
        {topicClusters.length > 0 && (
          <div className="card topic-clusters">
            <div className="card-header">
              <h3>Topic Distribution</h3>
            </div>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={topicClusters}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {topicClusters.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Backend Analytics */}
        {backendAnalytics && (
          <div className="card backend-analytics">
            <div className="card-header">
              <h3>🔧 System Analytics</h3>
            </div>
            <div className="analytics-grid">
              <div className="metric">
                <span className="metric-value">{backendAnalytics.total}</span>
                <span className="metric-label">Total Predictions</span>
              </div>
              <div className="metric">
                <span className="metric-value">{backendAnalytics.avgProcessingTime}ms</span>
                <span className="metric-label">Avg Processing Time</span>
              </div>
              <div className="metric">
                <span className="metric-value">{backendAnalytics.avgPrediction}%</span>
                <span className="metric-label">Avg Prediction</span>
              </div>
              <div className="metric">
                <span className="metric-value">{Math.round((backendAnalytics.successful / backendAnalytics.total) * 100)}%</span>
                <span className="metric-label">Success Rate</span>
              </div>
            </div>
          </div>
        )}

        {/* Prediction History */}
        {predictionHistory.length > 0 && (
          <div className="card history-panel">
            <div className="card-header">
              <h3>Recent Predictions</h3>
            </div>
            <div className="history-list">
              {predictionHistory.map((entry) => (
                <div key={entry.id} className="history-item">
                  <div className="history-text">{entry.text}</div>
                  <div className="history-meta">
                    <span className="history-prediction">{entry.prediction}%</span>
                    <span className="history-time">{entry.timestamp}</span>
                    <span className="history-words">{entry.wordCount} words</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-banner">
          <span className="error-icon">⚠️</span>
          <span className="error-text">{error}</span>
          <button onClick={() => setError(null)} className="error-close">✕</button>
        </div>
      )}

      {/* Status Indicators */}
      <div className="status-bar">
        <div className="status-item">
          <span className={`status-dot ${speechSupported ? 'green' : 'red'}`}></span>
          Speech Recognition: {speechSupported ? 'Available' : 'Not Available'}
        </div>
        <div className="status-item">
          <span className={`status-dot ${isListening ? 'red pulse' : 'gray'}`}></span>
          Microphone: {isListening ? 'Recording' : 'Idle'}
        </div>
        <div className="status-item">
          <span className={`status-dot ${backendAnalytics ? 'green' : 'gray'}`}></span>
          Backend: {backendAnalytics ? 'Connected' : 'Disconnected'}
        </div>
      </div>
    </div>
  );
}

export default App;