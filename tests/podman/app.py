"""
Simple Flask Hello World App for Podman Testing
"""
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Podman Test - Project Sanctuary</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                background: white;
                padding: 3rem;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                text-align: center;
                max-width: 500px;
            }
            h1 {
                color: #667eea;
                margin-bottom: 1rem;
            }
            .emoji {
                font-size: 4rem;
                margin: 1rem 0;
            }
            .info {
                background: #f0f4ff;
                padding: 1rem;
                border-radius: 10px;
                margin-top: 1rem;
            }
            .status {
                color: #10b981;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="emoji">🚀</div>
            <h1>Podman Test Successful!</h1>
            <p>Project Sanctuary - Task MCP Server</p>
            <div class="info">
                <p><span class="status">✅ Container Running</span></p>
                <p>Podman Desktop Integration: <strong>Working</strong></p>
                <p>Ready for MCP Server Deployment</p>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/health')
def health():
    return {'status': 'healthy', 'service': 'podman-test'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
