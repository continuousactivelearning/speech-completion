const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3001;

// Enhanced logging system
const logFile = path.join(__dirname, 'prediction_logs.json');

// Initialize log file if it doesn't exist
if (!fs.existsSync(logFile)) {
  fs.writeFileSync(logFile, JSON.stringify([]));
}

// Utility function to log predictions
const logPrediction = (data) => {
  try {
    const logs = JSON.parse(fs.readFileSync(logFile, 'utf8'));
    const logEntry = {
      timestamp: new Date().toISOString(),
      wordCount: data.transcript ? data.transcript.split(/\s+/).length : 0,
      textLength: data.transcript ? data.transcript.length : 0,
      prediction: data.prediction,
      processingTime: data.processingTime,
      success: data.success,
      error: data.error || null
    };
    
    logs.push(logEntry);
    
    if (logs.length > 100) {
      logs.splice(0, logs.length - 100);
    }
    
    fs.writeFileSync(logFile, JSON.stringify(logs, null, 2));
  } catch (error) {
    console.error('Failed to log prediction:', error);
  }
};

// CORS configuration
app.use(cors({
  origin: [
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'http://localhost:3002',  
    'http://127.0.0.1:3002'   
  ],
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
  credentials: true,
  preflightContinue: false,
  optionsSuccessStatus: 200
}));

app.options('/predict', cors());
app.use(bodyParser.json({ limit: '10mb' })); 

// Prediction endpoint
app.post('/predict', async (req, res) => {
  const startTime = Date.now();
  console.log("🛎️ Received request at /predict");

  const transcript = req.body.transcript;
  console.log(`Transcript received (${transcript ? transcript.length : 0} chars):`, 
              transcript ? transcript.substring(0, 100) + '...' : 'empty');

  if (!transcript || transcript.trim().length === 0) {
    const error = "Transcript is required and cannot be empty";
    logPrediction({ transcript, success: false, error, processingTime: Date.now() - startTime });
    return res.status(400).json({ error });
  }

  if (transcript.length > 50000) {
    const error = "Transcript too long. Maximum 50,000 characters allowed.";
    logPrediction({ transcript, success: false, error, processingTime: Date.now() - startTime });
    return res.status(400).json({ error });
  }

  const scriptPath = path.join(__dirname, '..', 'scripts', 'predict.py');
  console.log("Script path:", scriptPath);

  const python = spawn(
    path.join(__dirname, '..', 'scripts', 'venv', 'bin', 'python3'),
    [scriptPath]
  );

  python.stdin.write(JSON.stringify({ transcript }));
  python.stdin.end();

  let output = '';
  let errorOutput = '';

  python.stdout.on('data', (chunk) => {
    output += chunk.toString();
  });

  python.stderr.on('data', (err) => {
    errorOutput += err.toString();
    console.error("Python error:", err.toString());
  });

  python.on('close', (code) => {
    const processingTime = Date.now() - startTime;
    console.log(`Python process exited with code ${code} (${processingTime}ms)`);
    
    if (code !== 0) {
      console.error("Python script failed with code:", code);
      console.error("Error output:", errorOutput);
      
      logPrediction({ 
        transcript, 
        success: false, 
        error: `Python script error (code ${code})`, 
        processingTime 
      });
      
      return res.status(500).json({ 
        error: "Python script error", 
        details: errorOutput,
        code: code
      });
    }

    if (!output.trim()) {
      console.error("No output from Python script");
      logPrediction({ transcript, success: false, error: "No output from Python script", processingTime });
      return res.status(500).json({ error: "No output from Python script" });
    }

    try {
      const result = JSON.parse(output);
      console.log(`Result sent to frontend: ${result.prediction}% (${processingTime}ms)`);
      
      logPrediction({ 
        transcript, 
        prediction: result.prediction, 
        success: true, 
        processingTime 
      });
      
      res.json({ 
        prediction: result.prediction,
        metadata: {
          processingTime,
          wordCount: transcript.split(/\s+/).length,
          textLength: transcript.length,
          timestamp: new Date().toISOString()
        }
      });
      
    } catch (err) {
      console.error("Failed to parse Python output:", output);
      console.error("Parse error:", err.message);
      
      logPrediction({ 
        transcript, 
        success: false, 
        error: "Invalid Python response", 
        processingTime 
      });
      
      res.status(500).json({ 
        error: "Invalid Python response", 
        output: output.substring(0, 500)
      });
    }
  });

  python.on('error', (err) => {
    const processingTime = Date.now() - startTime;
    console.error("Failed to start Python process:", err);
    
    logPrediction({ 
      transcript, 
      success: false, 
      error: "Failed to start Python process", 
      processingTime 
    });
    
    res.status(500).json({ 
      error: "Failed to start Python process", 
      details: err.message 
    });
  });
});

// Analytics endpoint
app.get('/analytics', (req, res) => {
  try {
    const logs = JSON.parse(fs.readFileSync(logFile, 'utf8'));
    
    const analytics = {
      total: logs.length,
      successful: logs.filter(log => log.success).length,
      failed: logs.filter(log => !log.success).length,
      avgProcessingTime: logs.length > 0 ? 
        Math.round(logs.reduce((sum, log) => sum + log.processingTime, 0) / logs.length) : 0,
      avgPrediction: logs.filter(log => log.success && log.prediction).length > 0 ?
        Math.round(logs.filter(log => log.success && log.prediction)
          .reduce((sum, log) => sum + log.prediction, 0) / 
          logs.filter(log => log.success && log.prediction).length * 100) / 100 : 0,
      recentActivity: logs.slice(-10).reverse(),
      predictionDistribution: {
        early: logs.filter(log => log.success && log.prediction < 30).length,
        middle: logs.filter(log => log.success && log.prediction >= 30 && log.prediction < 70).length,
        late: logs.filter(log => log.success && log.prediction >= 70).length
      }
    };
    
    res.json(analytics);
  } catch (error) {
    console.error('Failed to read analytics:', error);
    res.status(500).json({ error: 'Failed to read analytics data' });
  }
});

// Health check
app.get('/health', (req, res) => {
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    node_version: process.version,
    environment: process.env.NODE_ENV || 'development'
  };
  res.json(health);
});

// Test endpoint
app.get('/test', (req, res) => {
  res.json({ 
    message: "Backend is working!", 
    timestamp: new Date().toISOString(),
    version: "2.0.0"
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({ 
    error: 'Internal server error',
    message: error.message 
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ 
    error: 'Endpoint not found',
    availableEndpoints: ['/test', '/predict', '/analytics', '/health']
  });
});

app.listen(PORT, () => {
  console.log(`Enhanced Speech Completion Server running on http://localhost:${PORT}`);
  console.log(`Analytics available at http://localhost:${PORT}/analytics`);
  console.log(`Health check at http://localhost:${PORT}/health`);
});
