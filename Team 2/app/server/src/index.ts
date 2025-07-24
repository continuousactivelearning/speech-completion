import dotenv from "dotenv";
dotenv.config();
import app from "./app";
import connectDB from "./db";

const PORT = process.env.PORT || 5000;

connectDB()
  .then(() => {
    app.on("error", (error) => {
      console.log("ERROR : ", error);
      throw error;
    });

    app.listen(PORT, () => {
      console.log(`Server is running at port : ${process.env.PORT}`);
    });
  })
  .catch((error) => {
    console.log("MONGODB connect failed : ", error);
  });
