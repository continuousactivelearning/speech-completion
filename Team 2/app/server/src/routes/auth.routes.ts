import { Router } from "express";
import { login, protectedRoute, signup } from "../controllers/auth.controller";
import { verifyToken } from "../middleware/auth.middleware";

const router = Router();

router.post("/signup", signup);
router.post("/login", login);
router.get("/protected", verifyToken, protectedRoute);

export default router;
