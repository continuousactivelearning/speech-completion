## Backend Folder Structure
```text
backend/
├── app.py
├── BiGRUTextNet_GloVe.py
├── completion_model.py
├── FineTuned_Gemma2B.py
├── glove.6B.300d.txt
├── inference_with_boosting.py
├── Is_Unique_Keyword.py
├── keyword_generator.py
├── knowledge_graph.py
├── main_script.py
├── model_completion.py
├── requirements.txt
├── summariser.py
├── gemma-2b-finetuned/
├── models/
│   └── BiGRUTextNet_GloVe.pt
```

## Downloads

Please download the following required files and place them **exactly as described**:

**Link:** [Download] ([https://drive.google.com/your_bigru_link](https://drive.google.com/drive/folders/1-PYsdyYRI51MYOMejVIsX0Cya-kyV0XK?usp=sharing)

### 1. GloVe Embeddings
- **File:** `glove.6B.300d.txt`
- **Place in:** `./backend`

### 2. BiGRU Trained Model
- **Folder:** `models/BiGRUTextNet_GloVe.pt`
- **Place in:** `./backend`

### 3. Fine-tuned Gemma 2B Model
- **Folder:**`gemma-2b-finetuned`
- **Place in:** `./backend`


## Setup
First, navigate to the `backend` directory and set up a virtual environment:

```bash
cd backend
python -m venv venv
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Running server (without ngrok)

```bash
python app.py
```

You shall find the server running on `http://127.0.0.1:5000`

## Running server (using ngrok)

Add the following lines of code to the flask file app.py
```bash
# Open the tunnel
public_url = ngrok.connect(5000)
print(f"🔗 Backend is live at: {public_url}")

# Start the server
app.run(port=5000)
```
You will find the terminal showing a URL `https://ngrok_your_url`
Put the same in the react file App.js in the folder `frontend`

