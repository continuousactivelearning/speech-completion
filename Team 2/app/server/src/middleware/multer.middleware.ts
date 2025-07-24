import multer from "multer";
import path from "path";

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, path.join(__dirname, "../../data"));
  },

  filename: function (req, file, cb) {
    if (file.mimetype === "text/csv" || file.originalname.endsWith(".csv")) {
      cb(null, "transcript.csv");
    } else {
      cb(null, Date.now() + "-" + file.originalname);
    }
  },
});

export const upload = multer({
  storage: storage,
  fileFilter: function (req, file, cb) {
    const allowedTypes = ["application.json", "text/csv"];

    if (
      !allowedTypes.includes(file.mimetype) &&
      !file.originalname.endsWith(".csv")
    ) {
      return cb(new Error("Only JSON or CSV files are allowed"));
    }
    cb(null, true);
  },
});
