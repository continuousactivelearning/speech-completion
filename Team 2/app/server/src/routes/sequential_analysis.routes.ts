import { Router } from "express";
import { initialPreprocessing } from "../controllers/sequentialAnalysis.controller";

const router = Router();

router.get("/initial_preprocessing", initialPreprocessing);

export default router;
