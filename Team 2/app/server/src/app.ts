import express, { Request, Response } from "express";
import cors from "cors";
import authRoutes from "./routes/auth.routes";
import transcriptRoutes from "./routes/transcript.routes";
import sequentialAnalysisRoutes from "./routes/sequential_analysis.routes";

const app = express();
app.use(
  cors({
    origin: "http://localhost:5173",
    credentials: true,
  })
);
app.use(express.json());

app.use("/api/v1/auth", authRoutes);
app.use("/api/v1/transcript", transcriptRoutes);
app.use("/api/v1/sequential_analysis", sequentialAnalysisRoutes);

app.get("/", (_: Request, res: Response) => {
  res.send("API Running");
});

export default app;
