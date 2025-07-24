# 📚 Transcript Topic Segmentation + Progress Labeling + Title Generation

This project was developed as part of a research internship, with a lot of experimentation around transcript analysis, topic segmentation, title generation, and more. Although the core goal was clear, the codebase grew with various experiments over time — this README will help you understand and run it locally.

---

## 🗂️ Project Structure (Important Folders Only)

### `app/`

Main web application:

- `/client`: Vite + React + TypeScript frontend
- `/server`: Express + TypeScript backend

### `python/`

All core ML/NLP experimentation and production scripts:

- Used to test and train various models.
- Contains a `Dockerfile` and `requirements.txt` for containerized setup.

---

## 🧠 Key Components Breakdown

### 🔧 Python Folder Overview

- `/data_collection_and_preprocessing`

  - `download_and_convert.py`: Downloads transcripts in bulk using `webvtt`, processes them.
  - `preprocess_transcript.py`: Legacy preprocessing, mostly replaced by the script above.
  - `titles.txt` & `video_links.txt`: Just paste your titles and links here before running the script.
  - `/transcript_csvs`: Generated transcript CSVs (ignore if already processed).

- `/clustering`

  - All clustering-related scripts and testing.
  - `/scripts`: Multiple trials with HDBSCAN, KMeans, Agglomerative, etc.
  - Settled on `HDBSCAN` + `all-MiniLM-L6-v2` from `sentence-transformers` for production.

- `/sequential_analysis`

  - Used only for initial exploratory analysis (e.g., `initial_preprocessing.ipynb`, `saturation_plot.ipynb`).

- `/topic_identification`

  - Contains experiments with various keyword extraction and title generation techniques:

    - `KeyBERT`, `TextRank`, `LDA`, `Top2Vec`, `BERTopic`, etc.
    - Also includes messy but insightful LLM title generation experiments.

- `/regressor_and_clustering`

  - `add_progress_label.py`: Adds `progress` label (0–1) to each chunk or bins it into classes depending on `MODE`.
  - Contains trials with regressors (Random Forest, Transformer, LSTM) and classifiers.
  - Final version uses LSTM Classifier.

- `/llm-check`

  - Tried out various LLMs (Gemma, Phi, Qwen, TinyLLaMA) to see which generates the best titles.

- `/ignore`

  - Just `check_proxy.py`, not relevant to core logic.

---

## 🌐 Web App (Frontend + Backend)

### `client/`

Standard Vite React setup (TypeScript):

- `components/`, `pages/`, `store/auth.ts`, `lib/axios.ts`
- Only minor setup needed to run it.

### `server/`

Express app for handling:

- Authentication (JWT)
- File upload
- Triggering Python scripts

Structure:

```
/__test__/mockUser.json     ← Used for testing locally (users should register)
/data                       ← Generated files stored here after Python runs
.env.sample                 ← Sample env file with JWT_SECRET, PORT, MONGO_URL
/docker-compose.yml         ← MongoDB setup for user storage
```

Inside `/src`:

- `controllers/`: Main ones are `auth.controller.ts`, `transcript.controller.ts`
- `middleware/`: `auth.middleware.ts` and `multer.middleware.ts`
- `routes/`: Corresponding to controllers
- `models/`: `user.model.ts`
- `utils/`: `generateToken.ts`
- `script/runPython.ts`: Triggers Python script from backend
- `index.ts` + `app.ts`: Main server entry

---

## 🧪 Server Python Scripts (`/app/server/python`)

Contains:

- `embed.py`: Used for generating embeddings
- `generate-titles.py`: Final title generation script
- `progressbar.py`: Adds progress labels using classifier
- `gain-curve.py`, `gain-curve-dump.py`: For measuring classifier confidence (via gain curves)
- `/embed_and_cluster/`: Old experiments, no longer in use
- `topic-title.pt`: Old model checkpoint, not used now

---

## 🛠️ Getting Started

### 1. Install Dependencies

**Backend:**

```bash
cd app/server
npm install
```

**Frontend:**

```bash
cd app/client
npm install
```

**Python:**

```bash
cd python
pip install -r requirements.txt
```

If you want to run inside Docker:

```bash
docker build -t transcript-analyzer .
```

### 2. MongoDB

You can use the provided `docker-compose.yml` in `/app/server` to spin up MongoDB:

```bash
docker-compose up -d
```

### 3. Environment Variables

Copy `.env.sample` in `/app/server` and rename it to `.env`. Add your JWT secret, MongoDB URI, etc.

```bash
cp .env.sample .env
```

---

## 🚀 Usage

1. Paste your YouTube links in `video_links.txt` and their titles in `titles.txt`.
2. Run:

   ```bash
   cd python/data_collection_and_preprocessing
   python download_and_convert.py
   ```

3. Go to the web app to start analyzing! (It will read from `/data`)

---

## 🚀 Project Tech Stack

### 💻 Frontend

- **Framework**: [Vite](https://vitejs.dev/) + [React 19](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- **Routing**: `react-router-dom`
- **Form Handling**: `react-hook-form`
- **State Management**: `zustand`
- **Charts**: `recharts`
- **UI**: `tailwindcss`, `lucide-react`

### 🌐 Backend (API Server)

- **Platform**: [Node.js](https://nodejs.org/) + [Express 5](https://expressjs.com/)
- **Language**: TypeScript
- **Database**: [MongoDB](https://www.mongodb.com/) via `mongoose`
- **Authentication**: `jsonwebtoken`, `bcryptjs`
- **File Uploads**: `multer`
- **Environment Management**: `dotenv`
- **Development Tools**: `nodemon`, `ts-node`

### 🧠 Machine Learning / NLP (Python)

- **Language**: Python
- **Clustering**: `HDBSCAN`
- **Embeddings**: `sentence-transformers`
- **Progress Estimation**: LSTM via `TensorFlow`/`Keras`
- **Feature Engineering & Modeling**: `scikit-learn`, `pandas`, `NumPy`
- **Visualization**: `matplotlib`, `seaborn`

### 🤖 LLM Integration (via [Ollama](https://ollama.com/))

- Tested with:

  - **Gemma**
  - **Qwen**
  - **TinyLLaMA**
  - **Phi-3**

---

## 🙏 Acknowledgements

This project was developed as part of a research internship. Huge thanks to the open-source libraries and pre-trained models that made this work possible.
