# Speech Completion Prediction - Method 1 (Janojit)

This repository contains code for preprocessing and evaluating speech completion data using transcript files. Follow the steps below to get started.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Team1-SpeechCompletion/Speech-Completion-Prediction.git
cd Speech-Completion-Prediction/method1_janojit
```

---

### 2. (Recommended) Create and Activate a Python Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

---

### 3. Install Dependencies

Make sure you have `pip` updated, then run:

```bash
pip install -r requirements.txt
```

---

### 4. Prepare Transcript Files

Create a folder named `transcripts` in the current directory and place all your `.txt` transcript files inside it:

```bash
mkdir transcripts
# Copy your .txt files into this folder
```

---

### 5. Label the Transcripts

Run the following script to label your transcripts:

```bash
python data_labelling.py
```

---

### 6. Split the Dataset

This script splits the dataset into training, testing, and validation sets. You can modify the split percentages in the script if needed.

```bash
python split_dataset.py
```

---

### 7. Download GloVe Embeddings

Before preprocessing, download the GloVe 300-dimensional word embeddings from [GloVe Website](https://nlp.stanford.edu/projects/glove/), specifically:

* `glove.6B.300d.txt`

Place the file in the root of the `method1_janojit/` directory:

```bash
# Example
mv glove.6B.300d.txt method1_janojit/
```

---

### 8. Preprocess the Completion Data

Run the preprocessing script:

```bash
python preprocess_completion_data.py
```

---

### 9. Evaluate the Model

Finally, run the evaluation script:

```bash
python evaluation.py
```

---

## 📂 Directory Structure

```
method1_janojit/
├── data_labelling.py
├── split_dataset.py
├── preprocess_completion_data.py
├── evaluation.py
├── requirements.txt
├── glove.6B.300d.txt
├── transcripts/
│   └── *.txt
└── ...
```

---

## 📝 Notes

* Ensure all transcript files are in `.txt` format.
* The scripts assume the presence of the `transcripts/` folder in the same directory.
* The GloVe file (`glove.6B.300d.txt`) is required for preprocessing.
* Modify any script as needed to fit your custom dataset or settings.

---
