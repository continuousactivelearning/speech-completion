import { spawn } from "child_process";
import path from "path";

export const runPython = (
  scriptName: string,
  args: string[] = []
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, "..", "..", "python", scriptName);
    const process = spawn("python", [scriptPath, ...args]);

    let result = "";
    let error = "";

    process.stdout.on("data", (data) => {
      result += data.toString();
    });

    process.stderr.on("data", (data) => {
      error += data.toString();
    });

    process.on("close", (code) => {
      if (code !== 0) {
        reject(error);
      } else {
        resolve(result.trim());
      }
    });
  });
};
