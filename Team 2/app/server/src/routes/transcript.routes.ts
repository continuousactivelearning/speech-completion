import { Router } from "express";
import { upload } from "../middleware/multer.middleware";
import {
  generateEmbeddingsAndClusters,
  generateLabels,
  uploadTranscript,
  generateTitles,
  getTitlesCsv,
  getPredictedProgress,
} from "../controllers/transcript.controller";

const router = Router();

router.post("/upload", upload.single("file"), uploadTranscript);
router.post("/embed_cluster", generateEmbeddingsAndClusters);
router.post("/add_progress_label", generateLabels);
router.post("/generate_titles", generateTitles);
router.get("/titles", getTitlesCsv);
router.get("/predict", getPredictedProgress);

export default router;
