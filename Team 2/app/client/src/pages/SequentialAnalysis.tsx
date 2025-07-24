import { useEffect, useState } from "react";
import { useAuth } from "../store/auth";
import { Navigate } from "react-router-dom";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

type GainPoint = {
  position: number;
  gain: number;
};

type GainFile = {
  text: string;
  data: GainPoint[];
};

type GainData = {
  [fileName: string]: GainFile;
};

export function SequentialAnalysis() {
  const { isAuthenticated } = useAuth();
  const [gainData, setGainData] = useState<GainData>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios
      .get(
        "http://localhost:5000/api/v1/sequential_analysis/initial_preprocessing",
        {
          withCredentials: true,
        }
      )
      .then((res) => {
        setGainData(res.data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error loading data", err);
        setLoading(false);
      });
  }, []);

  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  return (
    <div className="p-6 space-y-6 relative">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-bold">Sequential Gain Analysis</h1>
        <button
          onClick={() => useAuth.getState().logout()}
          className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg"
        >
          Logout
        </button>
      </div>

      {Object.entries(gainData).map(([fileName, fileData]) => (
        <div key={fileName} className="border rounded-xl p-4 shadow bg-white">
          <h2 className="font-semibold mb-2 text-purple-700">
            File: {fileName}
          </h2>
          <p className="mb-2 text-gray-600 text-sm">
            {fileData.text.substring(0, 200)}...
          </p>

          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={fileData.data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="position"
                label={{
                  value: "Word Position",
                  position: "insideBottom",
                  offset: -5,
                }}
              />
              <YAxis
                label={{ value: "Gain", angle: -90, position: "insideLeft" }}
              />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="gain"
                stroke="#7e22ce"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  );
}
