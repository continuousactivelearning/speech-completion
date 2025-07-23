## Frontend Setup (React + TailwindCSS)

### Folder Structure

```text
frontend/
├── build/
├── node_modules/
├── public/
├── src/
│   └── ... (your components and app files)
├── .gitignore
├── package-lock.json
├── package.json
├── README.md
```
## Prerequisites

Make sure you have Node.js and npm installed:

```bash
node -v
npm -v
```
If not installed, download from [https://nodejs.org.](https://nodejs.org.)

## Steps to Run the Frontend


### Install all dependencies:

```bash
cd frontend
npm install
```
### Set backend URL in `src/App.js`
```bash
//Ngrok URL if Ngrok is being used
const BACKEND_URL = 'https://2aa35a52bbd0.ngrok-free.app/'; // Update this if ngrok restarts
//for server running on local host, uncomment the line below 
//const BACKEND_URL ='http://127.0.0.1:5000';
```

### Start React app

```bash
npm start
```
This will start the development server at : `http://localhost:3000`
