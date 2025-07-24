import dotenv from "dotenv";
dotenv.config();

import jwt from "jsonwebtoken";

export const generateToken = (userId: string): string => {
  return jwt.sign({ userId }, process.env.JWT_SECRET!, {
    expiresIn: "1d",
  });
};
