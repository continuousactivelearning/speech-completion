import { useState } from "react";
import { Navigate } from "react-router-dom";
import {
  FileUp,
  UploadCloud,
  LogOut,
  FileText,
  LineChart as LineChartIcon,
  CheckCircle,
  Hourglass,
  PieChart,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import API from "../lib/axios";
import { useAuth } from "../store/auth";
import Button from "../components/Button";

type GainPoint = { position: number; gain: number };
type GainFile = { text: string; data: GainPoint[] };
type GainData = { [fileName: string]: GainFile };
type Topic = { start: string; title: string };

export default function Analysis() {
  const { isAuthenticated, logout } = useAuth();

  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [gainData, setGainData] = useState<GainData>({});
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);

  const [showGraph, setShowGraph] = useState(false);
  const [showTopics, setShowTopics] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [topics, setTopics] = useState<Topic[]>([]);

  const resetStateForNewUpload = () => {
    setShowGraph(false);
    setShowTopics(false);
    setUploadedFileName(null);
    setFile(null);
    setTopics([]);
  };

  const fetchGainData = async () => {
    try {
      setLoadingAnalysis(true);
      const res = await API.get("/sequential_analysis/initial_preprocessing", {
        withCredentials: true,
      });

      if (Object.keys(res.data).length > 0) {
        setGainData(res.data);
        const filename = Object.keys(res.data)[0];
        setUploadedFileName(filename);
        await fetchTopics();
        await fetchProgress();
      }
    } catch (err) {
      console.error("Error fetching analysis:", err);
    } finally {
      setLoadingAnalysis(false);
    }
  };

  const fetchTopics = async () => {
    try {
      const res = await API.get("/transcript/titles", {
        responseType: "text",
        withCredentials: true,
      });

      const lines = res.data.trim().split("\n").slice(1); // skip header
      const rows: Topic[] = lines.map((line: any) => {
        const [start, title] = line.split(",");
        return {
          start: start.trim(),
          title: title.trim(),
        };
      });

      setTopics(rows);
    } catch (err) {
      console.error("Error fetching topics:", err);
    }
  };

  const fetchProgress = async () => {
    try {
      const res = await API.get("/transcript/predict", {
        withCredentials: true,
      });

      const pct = Math.round(res.data.progress * 100);
      setProgress(pct);
    } catch (err) {
      console.error("Failed to fetch progress:", err);
      setProgress(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      resetStateForNewUpload();
      setFile(e.target.files[0]);
    }
  };

  const handleUploadAndAnalyze = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    setIsUploading(true);
    try {
      await API.post("/transcript/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      await fetchGainData();
      await fetchTopics();
      alert(`Successfully analyzed "${file.name}"!`);
    } catch (err: any) {
      console.error("Upload or Analysis failed:", err);
      alert("An error occurred: " + (err.message || "Please try again."));
      resetStateForNewUpload();
    } finally {
      setIsUploading(false);
    }
  };

  const handleGenerateGraph = () => setShowGraph((prev) => !prev);
  const handleShowTopics = () => setShowTopics((prev) => !prev);

  const handleDragEvents = {
    onDragEnter: (e: React.DragEvent<HTMLLabelElement>) => {
      e.preventDefault();
      setIsDragging(true);
    },
    onDragLeave: (e: React.DragEvent<HTMLLabelElement>) => {
      e.preventDefault();
      setIsDragging(false);
    },
    onDragOver: (e: React.DragEvent<HTMLLabelElement>) => {
      e.preventDefault();
    },
    onDrop: (e: React.DragEvent<HTMLLabelElement>) => {
      e.preventDefault();
      setIsDragging(false);
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        resetStateForNewUpload();
        setFile(e.dataTransfer.files[0]);
        e.dataTransfer.clearData();
      }
    },
  };

  if (!isAuthenticated) return <Navigate to="/login" />;

  const fileName = uploadedFileName || Object.keys(gainData)[0];
  const fileData: GainFile | null =
    fileName && gainData[fileName] ? gainData[fileName] : null;
  const isAnalysisReady = !!fileData;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-indigo-900 text-white font-sans p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        <header className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
              Analysis Dashboard
            </h1>
            <p className="text-sm text-indigo-300">
              A user-driven dashboard for transcript analysis.
            </p>
          </div>
          <Button
            onClick={logout}
            Icon={LogOut}
            className="bg-red-600 hover:bg-red-700"
          >
            Logout
          </Button>
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1 space-y-8">
            <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <UploadCloud size={22} /> Upload & Analyze
              </h3>
              <label
                htmlFor="dropzone-file"
                className={`flex flex-col items-center justify-center w-full h-40 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${
                  isDragging
                    ? "border-indigo-400 bg-white/20"
                    : "border-gray-500 hover:bg-white/10"
                }`}
                {...handleDragEvents}
              >
                <div className="flex flex-col items-center justify-center">
                  <FileUp className="w-8 h-8 mb-2 text-gray-400" />
                  <p className="text-sm text-gray-400 text-center">
                    <span className="font-semibold">Click to upload</span> or
                    drag & drop
                  </p>
                </div>
                <input
                  id="dropzone-file"
                  type="file"
                  className="hidden"
                  onChange={handleFileChange}
                  accept=".json,.csv"
                />
              </label>
              {file && (
                <div className="text-center mt-4">
                  <p className="text-sm text-indigo-300">
                    Selected: <span className="font-bold">{file.name}</span>
                  </p>
                  <Button
                    onClick={handleUploadAndAnalyze}
                    disabled={isUploading}
                    className="w-full mt-2"
                  >
                    {isUploading ? "Uploading..." : "Upload & Analyze"}
                  </Button>
                </div>
              )}
            </div>

            <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <PieChart size={22} /> View Results
              </h3>
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold text-gray-300">Status</h4>
                  {isAnalysisReady ? (
                    <>
                      <p className="text-green-400 flex items-center gap-2">
                        <CheckCircle size={18} /> Analysis ready for:{" "}
                        <span className="font-bold">{fileName}</span>
                      </p>

                      {/* ✅ Speech Progress Percentage Bar (Dummy) */}
                      <div className="mt-6">
                        <p className="text-sm text-indigo-300 mb-2">
                          Transcript Completion:
                        </p>
                        <div className="relative w-full bg-gray-800 rounded-full h-5 shadow-inner overflow-hidden">
                          <div
                            className="absolute left-0 top-0 h-full rounded-full bg-gradient-to-r from-green-400 via-blue-500 to-purple-600 transition-all duration-700"
                            style={{ width: `${progress ?? 0}%` }}
                          ></div>
                          <span
                            className="absolute right-2 top-1/2 transform -translate-y-1/2 text-xs font-bold text-white"
                            style={{ textShadow: "0 1px 2px rgba(0,0,0,0.6)" }}
                          >
                            {progress !== null ? `${progress}%` : "N/A"}
                          </span>
                        </div>
                      </div>
                    </>
                  ) : (
                    <p className="text-amber-400 flex items-center gap-2">
                      <Hourglass size={18} /> Waiting for file...
                    </p>
                  )}
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2.5">
                  <div
                    className="bg-indigo-500 h-2.5 rounded-full"
                    style={{ width: isAnalysisReady ? "100%" : "0%" }}
                  ></div>
                </div>
                <div className="pt-2 space-y-3">
                  <Button
                    onClick={handleGenerateGraph}
                    disabled={!isAnalysisReady}
                    className="w-full"
                  >
                    {showGraph ? "Hide Graph" : "Show Graph"}
                  </Button>
                  <Button
                    onClick={handleShowTopics}
                    disabled={!isAnalysisReady}
                    className="w-full bg-teal-600 hover:bg-teal-700"
                  >
                    {showTopics ? "Hide Topics" : "Show Topics"}
                  </Button>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-8 min-h-[400px]">
            {loadingAnalysis ? (
              <div className="flex items-center justify-center h-full bg-white/10 rounded-2xl p-10 text-center animate-pulse">
                <p className="text-lg text-gray-300">Loading analysis...</p>
              </div>
            ) : isAnalysisReady && fileData ? (
              <>
                {showGraph && (
                  <div className="bg-white/10 rounded-2xl p-6 animate-fade-in">
                    <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                      <LineChartIcon size={22} /> Gain Curve
                    </h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={fileData.data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
                        <XAxis dataKey="position" stroke="#A0AEC0" />
                        <YAxis stroke="#A0AEC0" />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "rgba(31, 41, 55, 0.8)",
                            borderColor: "#4A5568",
                            color: "#E5E7EB",
                            borderRadius: "0.5rem",
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="gain"
                          stroke="#8b5cf6"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {showTopics && (
                  <div className="bg-white/10 rounded-2xl p-6 animate-fade-in">
                    <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                      <FileText size={22} /> Topics & Timestamps
                    </h3>
                    <div className="bg-black/20 p-4 rounded-lg max-h-64 overflow-y-auto text-gray-300 text-sm leading-relaxed">
                      {topics.length === 0 ? (
                        <p className="text-gray-400 italic">No topics found.</p>
                      ) : (
                        topics.map((topic, idx) => (
                          <div key={idx} className="mb-4">
                            <p className="font-bold text-teal-400 mb-1">
                              [{topic.start}] - {topic.title}
                            </p>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="flex items-center justify-center h-full bg-white/10 rounded-2xl p-10 text-center animate-fade-in">
                <p className="text-lg text-gray-300">
                  Upload a file to begin analysis.
                </p>
              </div>
            )}
          </div>
        </main>
      </div>

      <style>{`
        .animate-fade-in {
          animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
