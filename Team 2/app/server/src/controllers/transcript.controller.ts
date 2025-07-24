import { Request, Response } from "express";
import path from "path";
import fs from "fs/promises";
import { runPython } from "../scripts/runPython";

export const uploadTranscript = async (req: Request, res: Response) => {
  try {
    if (!req.file) {
      res.status(400).json({ message: "No file uploaded" });
      return;
    }

    console.log("Uploaded file: ", req.file.filename);

    const inputCsv = path.join(__dirname, "../../data/transcript.csv");
    const embeddedCsv = path.join(__dirname, "../../data/embeddings.csv");
    // const clusteredCsv = path.join(
    //   __dirname,
    //   "../../data/clustered_embeddings.csv"
    // );
    const finalCsv = path.join(__dirname, "../../data/progress_added.csv");
    const titleCsv = path.join(__dirname, "../../data/titles.csv");

    // Step 1: embed and cluster
    // const clusterArgs = ["--input", inputCsv, "--output", clusteredCsv];
    // console.log("Running embeddings + clustering script after upload...");
    // const clusterResult = await runPython("embed_and_cluster.py", clusterArgs);
    const embedArgs = ["--input", inputCsv, "--output", embeddedCsv];
    console.log("Running embeddings script after upload...");
    const embedResult = await runPython("embed.py", embedArgs);
    console.log("Python cluster result:", embedResult);

    await fs.unlink(inputCsv);
    console.log("Deleted transcript.csv after embedding");

    // Step 2: progress labels
    const labelArgs = [
      "--input",
      embeddedCsv,
      "--output",
      finalCsv,
      "--num_classes",
      "20",
      "--mode",
      "classifier",
    ];
    console.log("Running progress labeling script...");
    const labelResult = await runPython("progressbar.py", labelArgs);
    console.log("Python label result:", labelResult);

    await fs.unlink(embeddedCsv);
    console.log("Deleted embeddings.csv after labeling");

    // Step 3: generate titles
    const titleArgs = ["--input", finalCsv, "--output", titleCsv];
    console.log("Generating titles...");
    const titleResult = await runPython("generate-titles.py", titleArgs);
    // console.log("Title generation result:", titleResult);

    res.status(200).json({
      message: "Upload + embeddings + labels + titles complete",
      filename: req.file.filename,
    });

    // Step 4: predict final progress
    const resultPath = path.join(__dirname, "../../data/final_result.txt");
    const predictArgs = ["--input", finalCsv, "--output", resultPath];
    console.log("Running final progress prediction...");
    const predictResult = await runPython("predict_progress.py", predictArgs);
    console.log("Final prediction result:", predictResult);
  } catch (e: any) {
    console.error("Upload error: ", e.message);
    res.status(500).json({ message: "Server error", details: e.message });
  }
};

export const generateEmbeddingsAndClusters = async (
  req: Request,
  res: Response
) => {
  try {
    const inputCsv = path.join(__dirname, "../../data/transcript.csv");
    const outputCsv = path.join(
      __dirname,
      "../../data/clustered_embeddings.csv"
    );

    const args = ["--input", inputCsv, "--output", outputCsv];

    console.log("Running embeddings + clustering script");
    const result = await runPython("embed_and_cluster.py", args);

    console.log("Python result:", result);

    const csvContent = await fs.readFile(outputCsv, "utf-8");

    res
      .status(200)
      .json({ message: "Embeddings + clustering done", csvContent });
  } catch (e: any) {
    console.error("Error in generateEmbeddingsAndClusters:", e.message);
    res.status(500).json({ message: "Server error", details: e.message });
  }
};

export const generateLabels = async (req: Request, res: Response) => {
  try {
    const mode = req.query.mode === "regression" ? "regression" : "classifier";
    const numClasses = parseInt(req.query.num_classes as string) || 20;

    const clusteredCsv = path.join(
      __dirname,
      "../../data/clustered_embeddings.csv"
    );
    const finalCsv = path.join(__dirname, "../../data/progress_added.csv");

    const args = [
      "--input",
      clusteredCsv,
      "--output",
      finalCsv,
      "--num_classes",
      numClasses.toString(),
      "--mode",
      mode,
    ];

    console.log("Running progress labeling...");
    const result = await runPython("progressbar.py", args);
    console.log("Labeling result:", result);

    const csvContent = await fs.readFile(finalCsv, "utf-8");

    res.status(200).json({
      message: "Progress labels generated",
      mode,
      num_classes: numClasses,
      csvContent,
    });
  } catch (e: any) {
    console.error("Error in generateLabels:", e.message);
    res.status(500).json({ message: "Server error", details: e.message });
  }
};

// manual title generation
export const generateTitles = async (req: Request, res: Response) => {
  try {
    const inputCsv = path.join(
      __dirname,
      "../../data/clustered_embeddings.csv"
    );
    const outputCsv = path.join(__dirname, "../../data/titles.csv");

    const args = ["--input", inputCsv, "--output", outputCsv];

    console.log("Running title generation...");
    const result = await runPython("generate-titles.py", args);
    console.log("Title generation result:", result);

    const csvContent = await fs.readFile(outputCsv, "utf-8");

    res.status(200).json({
      message: "Titles generated",
      csvContent,
    });
  } catch (e: any) {
    console.error("Error in generateTitles:", e.message);
    res.status(500).json({ message: "Server error", details: e.message });
  }
};

export const getTitlesCsv = async (req: Request, res: Response) => {
  try {
    const filePath = path.join(__dirname, "../../data/titles.csv");
    const csvContent = await fs.readFile(filePath, "utf-8");
    res.type("text/csv").send(csvContent);
  } catch (e: any) {
    console.error("Error reading titles.csv:", e.message);
    res
      .status(500)
      .json({ message: "Failed to load titles", error: e.message });
  }
};

// manual progress  prediction
export const getPredictedProgress = async (req: Request, res: Response) => {
  try {
    const filePath = path.join(__dirname, "../../data/final_result.txt");
    const raw = await fs.readFile(filePath, "utf-8");
    const progress = parseFloat(raw);
    res.status(200).json({ progress });
  } catch (e: any) {
    console.error("Error reading final_result.txt:", e.message);
    res
      .status(500)
      .json({ message: "Failed to read progress result", error: e.message });
  }
};
