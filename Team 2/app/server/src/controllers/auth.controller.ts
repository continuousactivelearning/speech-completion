import { Request, Response } from "express";
import bcrypt from "bcryptjs";
import { generateToken } from "../utils/generateToken";
import User from "../models/user.model";

export const signup = async (req: Request, res: Response) => {
  const { username, email, password } = req.body;
  try {
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      throw new Error("User already exists");
    }
    const hashed = await bcrypt.hash(password, 10);
    const user = await User.create({
      username: username,
      email: email,
      password: hashed,
    });

    const token = generateToken(user._id.toString());
    res.status(200).json({
      token,
      user: {
        _id: user._id,
        username: user.username,
        email: user.email,
      },
    });
  } catch (e: any) {
    res.status(400).json({ message: "Signup failed", details: e.message });
  }
};

export const login = async (req: Request, res: Response) => {
  const { email, password } = req.body;
  try {
    const user = await User.findOne({ email });
    if (!user) throw new Error("Invalid credentials");
    const match = await bcrypt.compare(password, user.password);
    if (!match) throw new Error("Invalid password");
    const token = generateToken(user._id.toString());
    res.status(200).json({
      token,
      user: {
        _id: user._id,
        username: user.username,
        email: user.email,
      },
    });
  } catch (e: any) {
    res.status(401).json({ message: "Signin failed", details: e.message });
  }
};

export const protectedRoute = (req: Request, res: Response) => {
  res.status(200).json({ message: "You have accessed a protected route" });
};
