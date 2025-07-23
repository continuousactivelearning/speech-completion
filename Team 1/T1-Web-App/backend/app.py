from flask import Flask, request, jsonify
from flask_cors import CORS
from main_script import analyze, reset  # Make sure both functions are defined

from pyngrok import ngrok

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_route():
    return analyze()

@app.route('/reset', methods=['POST'])
def reset_route():
    return reset()
    
'''Uncomment while using ngrok
# 🔗 Open the tunnel
public_url = ngrok.connect(5000)
print(f"🔗 Backend is live at: {public_url}")

# 🚀 Start the server
app.run(port=5000)
'''
# Comment the code below if using ngrok
if __name__ == '__main__':
    app.run(debug=True)
