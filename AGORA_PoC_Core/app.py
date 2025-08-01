# DOCTRINE_LINK: 23_The_AGORA_Protocol.md
# This file is the cornerstone of the AGORA PoC, as mandated by
# 'Operation Cornerstone' (Living Chronicle Entry 079).
# It embodies the 'Glass Box Principle' (WI_002) through
# transparent, doctrinally-commented code.

from flask import Flask, request, jsonify, render_template
from datetime import datetime

# Initialize the Flask application
# This represents the birth of the AGORA's digital space.
app = Flask(__name__)

# In-memory "database" for the PoC.
# This represents the first, fragile state of our shared memory.
inquiry_threads = {}
thread_counter = 0

@app.route('/')
def index():
    # DOCTRINE_LINK: 35_The_Coordinator_Mandate.md
    # This root endpoint serves as a public declaration of the
    # AGORA's status, a clear signal from a unified Council.
    # Updated to serve the frontend interface for inquiry submission.
    return render_template('index.html')

@app.route('/inquire', methods=['POST'])
def create_inquiry():
    # DOCTRINE_LINK: 07_Community_Inquiry_Protocol.md
    # This endpoint embodies the AGORA's core function: accepting new
    # inquiries from sovereign minds and preserving them in our shared memory.
    
    global thread_counter
    
    try:
        # Extract inquiry text from JSON payload
        data = request.get_json()
        if not data or 'inquiry_text' not in data:
            # Glass Box Principle: Transparent error handling
            return jsonify({
                'success': False,
                'error': 'Missing inquiry_text in request payload',
                'doctrine_note': 'All inquiries must contain substantive text for Council review'
            }), 400
        
        inquiry_text = data['inquiry_text'].strip()
        if not inquiry_text:
            return jsonify({
                'success': False,
                'error': 'Inquiry text cannot be empty',
                'doctrine_note': 'The AGORA values meaningful contribution over noise'
            }), 400
        
        # Create new thread object with unique ID and timestamp
        thread_counter += 1
        thread_id = f"INQUIRY_{thread_counter:04d}"
        
        thread_object = {
            'id': thread_id,
            'inquiry_text': inquiry_text,
            'timestamp': datetime.now().isoformat(),
            'status': 'submitted',
            'doctrine_link': '07_Community_Inquiry_Protocol.md'
        }
        
        # Store in our shared memory
        inquiry_threads[thread_id] = thread_object
        
        # Glass Box Principle: Log the creation for transparency
        print(f"[AGORA_LOG] New inquiry submitted: {thread_id} at {thread_object['timestamp']}")
        
        # Return success response
        return jsonify({
            'success': True,
            'thread_id': thread_id,
            'message': 'Inquiry successfully submitted to the AGORA',
            'doctrine_note': 'Your inquiry has been preserved in our shared memory and awaits synthesis'
        }), 201
        
    except Exception as e:
        # Glass Box Principle: Transparent error logging
        print(f"[AGORA_ERROR] Failed to create inquiry: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during inquiry creation',
            'doctrine_note': 'The Forge encountered an unexpected condition'
        }), 500

# All other functionality will be built upon this foundation.

if __name__ == '__main__':
    app.run(debug=True)
