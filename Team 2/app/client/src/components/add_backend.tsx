import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { PieChart, Pie, Cell } from 'recharts';
import { FileUp, LogOut, BarChart2, List, Settings, UploadCloud } from 'lucide-react';

// Mock Data for the graph until backend is connected
const generateMockGainData = () => {
    const data = [];
    let gain = 0;
    for (let i = 0; i < 100; i++) {
        gain += (Math.random() - 0.5) * 0.5;
        data.push({ position: i, gain: parseFloat(gain.toFixed(2)) });
    }
    return [{ text: "Mock Transcript Snippet...", data }];
};

const mockTopics = [
    { title: "Introduction to Quantum Physics", timestamp: "00:01:23" },
    { title: "Heisenberg's Uncertainty Principle", timestamp: "00:15:45" },
    { title: "Quantum Entanglement", timestamp: "00:28:10" },
    { title: "Schrödinger's Cat Paradox", timestamp: "00:42:55" },
];

// Reusable Components
const Card = ({ children, className = '' }) => (
    <div className={`bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl shadow-lg p-6 ${className}`}>
        {children}
    </div>
);

const Button = ({ children, onClick, className = '', Icon, disabled = false }) => (
    <button
        onClick={onClick}
        disabled={disabled}
        className={`flex items-center justify-center gap-2 px-4 py-2 bg-indigo-600 text-white font-semibold rounded-lg shadow-md hover:bg-indigo-700 transition-all duration-300 disabled:bg-gray-500 disabled:cursor-not-allowed ${className}`}
    >
        {Icon && <Icon size={18} />}
        {children}
    </button>
);

const Input = ({ label, type = 'text', value, onChange, placeholder }) => (
    <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
        <input
            type={type}
            value={value}
            onChange={onChange}
            placeholder={placeholder}
            className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg focus:ring-2 focus:ring-indigo-400 focus:outline-none transition-all duration-300"
        />
    </div>
);


// --- Authentication Component ---
const AuthPage = ({ onAuth }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [name, setName] = useState('');
    const [error, setError] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        setError('');
        if (!email || !password || (!isLogin && !name)) {
            setError("Please fill in all fields.");
            return;
        }
        // Mock authentication
        console.log(`${isLogin ? 'Logging in' : 'Signing up'} with:`, { name, email });
        onAuth({ name: isLogin ? 'Demo User' : name, email });
    };

    return (
        <div className="min-h-screen w-full flex items-center justify-center bg-gradient-to-br from-gray-900 to-indigo-900 text-white font-sans p-4">
            <Card className="w-full max-w-md">
                <h2 className="text-3xl font-bold text-center mb-2">{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
                <p className="text-center text-gray-400 mb-6">{isLogin ? 'Sign in to continue' : 'Get started with your analysis'}</p>

                {error && <p className="bg-red-500/50 text-white p-3 rounded-lg text-center mb-4">{error}</p>}

                <form onSubmit={handleSubmit} className="space-y-4">
                    {!isLogin && <Input label="Full Name" value={name} onChange={(e) => setName(e.target.value)} placeholder="John Doe" />}
                    <Input label="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
                    <Input label="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" />
                    <Button type="submit" className="w-full !py-3 !text-base">
                        {isLogin ? 'Sign In' : 'Sign Up'}
                    </Button>
                </form>

                <p className="text-sm text-center mt-6 text-gray-400">
                    {isLogin ? "Don't have an account?" : 'Already have an account?'}
                    <button onClick={() => setIsLogin(!isLogin)} className="font-semibold text-indigo-400 hover:underline ml-1">
                        {isLogin ? 'Sign Up' : 'Sign In'}
                    </button>
                </p>
            </Card>
        </div>
    );
};


// --- Dashboard Component ---
const Dashboard = ({ user, onLogout }) => {
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [analysisReady, setAnalysisReady] = useState(false);
    const [showGraph, setShowGraph] = useState(false);
    const [showTopics, setShowTopics] = useState(false);
    const [gainData, setGainData] = useState([]);
    const [topics, setTopics] = useState([]);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile) {
            setFile(selectedFile);
            setAnalysisReady(false); // Reset analysis state on new file selection
            setShowGraph(false);
            setShowTopics(false);
        }
    };

    const handleUpload = () => {
        if (!file) return;
        setUploading(true);
        // Mock upload process
        setTimeout(() => {
            setUploading(false);
            setAnalysisReady(true);
        }, 1500);
    };

    const handleGenerateAnalysis = () => {
        // Mock data fetching
        setShowGraph(true);
        setGainData(generateMockGainData());
    };

    const handleShowTopics = () => {
        setShowTopics(true);
        setTopics(mockTopics);
    };

    const PIE_DATA = [
      { name: 'Analysis', value: 70, color: '#8b5cf6' },
      { name: 'Review', value: 30, color: '#a78bfa' },
    ];

    return (
        <div className="min-h-screen w-full bg-gradient-to-br from-gray-900 to-indigo-900 text-white font-sans p-4 sm:p-6 lg:p-8">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <header className="flex justify-between items-center mb-8">
                    <div>
                        <h1 className="text-2xl sm:text-3xl font-bold">Transcript Dashboard</h1>
                        <p className="text-gray-400">Welcome back, {user.name}!</p>
                    </div>
                    <div className="flex items-center gap-4">
                        <Button onClick={onLogout} Icon={LogOut} className="bg-red-600 hover:bg-red-700">Logout</Button>
                        <Settings className="text-gray-400 hover:text-white cursor-pointer transition-colors"/>
                    </div>
                </header>

                <main className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Left Column: Controls */}
                    <div className="lg:col-span-1 space-y-8">
                        <Card>
                            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2"><UploadCloud size={22}/> Upload Transcript</h3>
                            <div className="flex flex-col items-center justify-center w-full">
                                <label htmlFor="dropzone-file" className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-gray-500 rounded-lg cursor-pointer hover:bg-white/10 transition-colors">
                                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                        <FileUp className="w-10 h-10 mb-3 text-gray-400" />
                                        <p className="mb-2 text-sm text-gray-400 text-center"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                                        <p className="text-xs text-gray-500">TXT, CSV or other text formats</p>
                                    </div>
                                    <input id="dropzone-file" type="file" className="hidden" onChange={handleFileChange} accept=".txt,.csv" />
                                </label>
                            </div>
                            {file && <p className="mt-4 text-center text-sm text-indigo-300">Selected: {file.name}</p>}

                            <Button onClick={handleUpload} disabled={!file || uploading || analysisReady} className="w-full mt-4">
                                {uploading ? 'Uploading...' : 'Upload'}
                            </Button>
                        </Card>

                        {analysisReady && (
                            <Card className="animate-fade-in">
                                <h3 className="text-xl font-semibold mb-4 flex items-center gap-2"><BarChart2 size={22}/> Analysis Options</h3>
                                <div className="space-y-3">
                                  <Button onClick={handleGenerateAnalysis} Icon={BarChart2} className="w-full bg-green-600 hover:bg-green-700">
                                      Show Sequential Analysis
                                  </Button>
                                  <Button onClick={handleShowTopics} Icon={List} className="w-full bg-blue-600 hover:bg-blue-700">
                                      Show Topics & Timestamps
                                  </Button>
                                </div>
                            </Card>
                        )}
                        
                         <Card>
                            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">Project Status</h3>
                            <div style={{width: '100%', height: 150}}>
                                <ResponsiveContainer>
                                    <PieChart>
                                        <Pie data={PIE_DATA} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={60} fill="#8884d8" label>
                                            {PIE_DATA.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.color} />)}
                                        </Pie>
                                        <Tooltip />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                             <p className="text-center text-gray-400 mt-2">70% of transcript analyzed.</p>
                        </Card>
                    </div>

                    {/* Right Column: Display Area */}
                    <div className="lg:col-span-2">
                        <Card className="min-h-[30rem] lg:min-h-full flex flex-col justify-center items-center animate-fade-in">
                            {!showGraph && !showTopics && (
                                <div className="text-center text-gray-400">
                                    <BarChart2 size={48} className="mx-auto mb-4"/>
                                    <h3 className="text-xl font-semibold">Your analysis will appear here.</h3>
                                    <p>Upload a transcript and select an analysis option to begin.</p>
                                </div>
                            )}

                            {showGraph && gainData.length > 0 && (
                                <div className="w-full h-full animate-fade-in">
                                    <h3 className="text-xl font-semibold mb-4 text-center">Sequential Analysis Curve</h3>
                                    <div style={{width: '100%', height: 400}}>
                                        <ResponsiveContainer>
                                            <LineChart data={gainData[0].data} margin={{ top: 5, right: 20, left: -10, bottom: 20 }}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
                                                <XAxis dataKey="position" stroke="#A0AEC0" label={{ value: 'Word Position', position: 'insideBottom', offset: -15, fill: '#A0AEC0' }} />
                                                <YAxis stroke="#A0AEC0" label={{ value: 'Gain', angle: -90, position: 'insideLeft', fill: '#A0AEC0' }} />
                                                <Tooltip
                                                    contentStyle={{
                                                        backgroundColor: 'rgba(31, 41, 55, 0.8)',
                                                        borderColor: '#4A5568',
                                                        color: '#E5E7EB'
                                                    }}
                                                />
                                                <Legend wrapperStyle={{ color: '#E5E7EB' }}/>
                                                <Line type="monotone" dataKey="gain" stroke="#6366f1" strokeWidth={2} dot={false} />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            )}

                            {showTopics && topics.length > 0 && (
                                <div className="w-full h-full animate-fade-in">
                                    <h3 className="text-xl font-semibold mb-4 text-center">Topics & Timestamps</h3>
                                    <ul className="space-y-3">
                                        {topics.map((topic, index) => (
                                            <li key={index} className="bg-white/5 p-4 rounded-lg flex justify-between items-center transition hover:bg-white/10">
                                                <span className="font-medium">{topic.title}</span>
                                                <span className="text-sm text-indigo-300 font-mono bg-indigo-900/50 px-2 py-1 rounded">{topic.timestamp}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </Card>
                    </div>
                </main>
            </div>
             <style jsx global>{`
                .animate-fade-in {
                    animation: fadeIn 0.5s ease-in-out;
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            `}</style>
        </div>
    );
};


// --- Main App Component ---
export default function App() {
    const [user, setUser] = useState(null);

    // This effect checks for a logged-in user, e.g., from localStorage
    useEffect(() => {
        // const storedUser = localStorage.getItem('user');
        // if (storedUser) {
        //     setUser(JSON.parse(storedUser));
        // }
    }, []);


    const handleAuth = (authenticatedUser) => {
        setUser(authenticatedUser);
        // localStorage.setItem('user', JSON.stringify(authenticatedUser));
    };

    const handleLogout = () => {
        setUser(null);
        // localStorage.removeItem('user');
    };

    // Use a simple conditional render instead of a router
    if (!user) {
        return <AuthPage onAuth={handleAuth} />;
    }

    return <Dashboard user={user} onLogout={handleLogout} />;
}
