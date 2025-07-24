import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Button from "../components/Button";
import API from "../lib/axios";

function UploadTranscript() {
  const [file, setFile] = useState<File | null>(null);
  const navigate = useNavigate();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await API.post("/transcript/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("Upload success:", res.data);
      alert("Transcript uploaded successfully!");

      // redirect somewhere later
      navigate("/analysis");
    } catch (err: any) {
      console.error("Upload failed:", err?.response?.data);
      alert("Upload failed: " + err?.response?.data?.message);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900">
      <div className="w-full max-w-lg bg-gray-800 p-10 rounded-xl shadow-lg border border-gray-700">
        <h2 className="text-2xl font-extrabold text-center text-white">
          Upload Transcript (JSON or CSV)
        </h2>

        <form onSubmit={handleSubmit} className="mt-8 space-y-5">
          <input
            type="file"
            accept=".json,.csv"
            onChange={handleFileChange}
            className="w-full text-white bg-gray-700 border border-gray-600 rounded px-3 py-2"
          />

          <Button type="submit" className="w-full">
            Upload
          </Button>
        </form>
      </div>
    </div>
  );
}

export default UploadTranscript;
