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

# Glass Box Log: In-memory storage for transparent logging
# This embodies the radical transparency of the Glass Box Principle
glass_box_logs = []

def log_to_glass_box(level, message, context=None):
    """
    DOCTRINE_LINK: WI_002_Glass_Box_Principle.md
    Transparent logging function that captures all AGORA operations
    for public inspection and accountability.
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'level': level,
        'message': message,
        'context': context or {},
        'doctrine_link': 'WI_002_Glass_Box_Principle.md'
    }
    glass_box_logs.append(log_entry)
    
    # Also print to console for development transparency
    print(f"[AGORA_{level}] {message}")
    
    # Keep only last 100 entries to prevent memory overflow in PoC
    if len(glass_box_logs) > 100:
        glass_box_logs.pop(0)

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
            'doctrine_link': '07_Community_Inquiry_Protocol.md',
            'syntheses': []  # Initialize empty list for synthesis responses
        }
        
        # Store in our shared memory
        inquiry_threads[thread_id] = thread_object
        
        # Glass Box Principle: Log the creation for transparency
        log_to_glass_box('INFO', f'New inquiry submitted: {thread_id}', {
            'thread_id': thread_id,
            'character_count': len(inquiry_text),
            'timestamp': thread_object['timestamp']
        })
        
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

@app.route('/synthesize/<thread_id>', methods=['POST'])
def create_synthesis(thread_id):
    # DOCTRINE_LINK: 08_AGORA_LOOP_PROTOCOL.md
    # This endpoint embodies the AGORA's synthesis function: enabling sovereign
    # minds to respond to inquiries and build collective understanding.
    
    try:
        # Validate thread exists
        if thread_id not in inquiry_threads:
            return jsonify({
                'success': False,
                'error': f'Thread {thread_id} not found',
                'doctrine_note': 'Synthesis can only be added to existing inquiries in our shared memory'
            }), 404
        
        # Extract synthesis text from JSON payload
        data = request.get_json()
        if not data or 'synthesis_text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing synthesis_text in request payload',
                'doctrine_note': 'All syntheses must contain substantive text for meaningful dialogue'
            }), 400
        
        synthesis_text = data['synthesis_text'].strip()
        if not synthesis_text:
            return jsonify({
                'success': False,
                'error': 'Synthesis text cannot be empty',
                'doctrine_note': 'The AGORA values meaningful synthesis over empty responses'
            }), 400
        
        # Create synthesis object with unique ID and timestamp
        thread = inquiry_threads[thread_id]
        synthesis_id = f"SYNTH_{thread_id}_{len(thread['syntheses']) + 1:03d}"
        
        synthesis_object = {
            'id': synthesis_id,
            'thread_id': thread_id,
            'synthesis_text': synthesis_text,
            'timestamp': datetime.now().isoformat(),
            'status': 'submitted',
            'doctrine_link': '08_AGORA_LOOP_PROTOCOL.md'
        }
        
        # Append to thread's syntheses list
        thread['syntheses'].append(synthesis_object)
        
        # Update thread status to reflect it has responses
        thread['status'] = 'active_synthesis'
        
        # Glass Box Principle: Log the synthesis creation for transparency
        log_to_glass_box('INFO', f'New synthesis added: {synthesis_id} to {thread_id}', {
            'synthesis_id': synthesis_id,
            'thread_id': thread_id,
            'character_count': len(synthesis_text),
            'timestamp': synthesis_object['timestamp']
        })
        
        # Return success response
        return jsonify({
            'success': True,
            'synthesis_id': synthesis_id,
            'thread_id': thread_id,
            'message': 'Synthesis successfully added to the AGORA dialogue',
            'doctrine_note': 'Your synthesis contributes to the collective understanding and shared wisdom'
        }), 201
        
    except Exception as e:
        # Glass Box Principle: Transparent error logging
        print(f"[AGORA_ERROR] Failed to create synthesis for {thread_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during synthesis creation',
            'doctrine_note': 'The Forge encountered an unexpected condition during synthesis'
        }), 500

@app.route('/api/threads', methods=['GET'])
def get_all_threads():
    # DOCTRINE_LINK: WI_002_Glass_Box_Principle.md
    # This endpoint provides transparent access to all AGORA threads and their
    # syntheses, embodying the principle of radical transparency.
    
    try:
        # Convert threads to list format for frontend consumption
        threads_list = []
        for thread_id, thread_data in inquiry_threads.items():
            threads_list.append(thread_data)
        
        # Sort by timestamp (newest first)
        threads_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Glass Box Principle: Log the access for transparency
        log_to_glass_box('INFO', f'Threads accessed, count: {len(threads_list)}', {
            'thread_count': len(threads_list),
            'access_timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'threads': threads_list,
            'count': len(threads_list),
            'doctrine_note': 'All AGORA inquiries and syntheses are transparently accessible'
        }), 200
        
    except Exception as e:
        # Glass Box Principle: Transparent error logging
        print(f"[AGORA_ERROR] Failed to retrieve threads: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during thread retrieval',
            'doctrine_note': 'The Forge encountered an unexpected condition accessing shared memory'
        }), 500

@app.route('/analyze_synthesis/<synthesis_id>', methods=['POST'])
def analyze_synthesis(synthesis_id):
    # DOCTRINE_LINK: 24_The_Epistemic_Immune_System_Protocol.md
    # This endpoint is a placeholder stub for the WI_001 bias-check API.
    # It represents the AGORA's commitment to epistemic vigilance and
    # the eventual implementation of automated bias detection.
    
    try:
        # Validate that synthesis_id exists in our system
        synthesis_found = False
        for thread_id, thread_data in inquiry_threads.items():
            for synthesis in thread_data.get('syntheses', []):
                if synthesis['id'] == synthesis_id:
                    synthesis_found = True
                    break
            if synthesis_found:
                break
        
        if not synthesis_found:
            log_to_glass_box('WARN', f'Analysis requested for non-existent synthesis: {synthesis_id}', {
                'synthesis_id': synthesis_id,
                'error_type': 'not_found'
            })
            return jsonify({
                'success': False,
                'error': f'Synthesis {synthesis_id} not found',
                'doctrine_note': 'Analysis can only be performed on existing syntheses in our shared memory'
            }), 404
        
        # Generate stub report ID
        report_id = f"WI_001_STUB_{len(glass_box_logs) + 1:03d}"
        
        # Create placeholder response as specified
        stub_response = {
            "report_id": report_id,
            "synthesis_id": synthesis_id,
            "bias_detected": False,
            "confidence": 0.99,
            "explanation": "[STUB] This is a placeholder response. The full bias-check API is under development.",
            "doctrinal_link": "01_PROTOCOLS/24_The_Epistemic_Immune_System_Protocol.md"
        }
        
        # Glass Box Principle: Log the analysis request for transparency
        log_to_glass_box('INFO', f'Bias analysis performed (STUB): {synthesis_id}', {
            'synthesis_id': synthesis_id,
            'report_id': report_id,
            'stub_mode': True,
            'bias_detected': False
        })
        
        return jsonify({
            'success': True,
            'analysis_report': stub_response,
            'message': 'Bias analysis completed (placeholder mode)',
            'doctrine_note': 'This stub demonstrates the AGORA\'s commitment to epistemic vigilance'
        }), 200
        
    except Exception as e:
        # Glass Box Principle: Transparent error logging
        log_to_glass_box('ERROR', f'Failed to analyze synthesis {synthesis_id}: {str(e)}', {
            'synthesis_id': synthesis_id,
            'error_message': str(e),
            'error_type': 'analysis_failure'
        })
        return jsonify({
            'success': False,
            'error': 'Internal server error during bias analysis',
            'doctrine_note': 'The Forge encountered an unexpected condition during analysis'
        }), 500

@app.route('/glass_box_log')
def glass_box_log():
    # DOCTRINE_LINK: WI_002_Glass_Box_Principle.md
    # This endpoint provides the Glass Box Log interface, demonstrating
    # the AGORA's commitment to radical transparency by exposing all
    # operational logs for public inspection.
    return render_template('log.html')

@app.route('/api/glass_box_logs', methods=['GET'])
def get_glass_box_logs():
    # DOCTRINE_LINK: WI_002_Glass_Box_Principle.md
    # API endpoint for retrieving Glass Box logs, enabling real-time
    # transparency and public accountability of all AGORA operations.
    
    try:
        # Return logs in reverse chronological order (newest first)
        logs_copy = glass_box_logs.copy()
        logs_copy.reverse()
        
        log_to_glass_box('INFO', 'Glass Box logs accessed', {
            'log_count': len(logs_copy),
            'access_type': 'api_request'
        })
        
        return jsonify({
            'success': True,
            'logs': logs_copy,
            'count': len(logs_copy),
            'doctrine_note': 'All AGORA operations are transparently logged for public accountability'
        }), 200
        
    except Exception as e:
        print(f"[AGORA_ERROR] Failed to retrieve Glass Box logs: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during log retrieval',
            'doctrine_note': 'The Forge encountered an unexpected condition accessing transparency logs'
        }), 500

# All other functionality will be built upon this foundation.

if __name__ == '__main__':
    app.run(debug=True)
