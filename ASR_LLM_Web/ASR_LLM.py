from flask import Flask, render_template, request, jsonify
import openai
import threading
import queue
import tempfile
import os
import sounddevice as sd
import wave

# Initialize Flask server
app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = ""

# Queue to manage ASR processing
asr_queue = queue.Queue()

# Function to handle ASR processing
def asr_worker():
    while True:
        audio_file_path, res_queue = asr_queue.get()
        if audio_file_path is None:
            break
        try:
            # Use OpenAI Whisper API to transcribe the audio
            with open(audio_file_path, "rb") as audio_file:
                result = openai.Audio.transcribe("whisper-1", audio_file)
                recognized_text = result['text']
            res_queue.put(recognized_text)
        except Exception as e:
            res_queue.put(f"Error: {str(e)}")

# Start a separate thread for ASR processing
asr_thread = threading.Thread(target=asr_worker, daemon=True)
asr_thread.start()

# Route to serve the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to start recording from microphone
@app.route('/record', methods=['POST'])
def record():
    duration = 5  # Recording duration in seconds
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    filename = temp_audio.name
    temp_audio.close()
    fs = 44100  # Sample rate
    
    # Record audio
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    
    # Save the recording to a file
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())
    
    # Add file to ASR queue
    res_queue = queue.Queue()
    asr_queue.put((filename, res_queue))
    recognized_text = res_queue.get()
    try:
        os.unlink(filename)
    except FileNotFoundError:
        pass  # Delete the temporary file
    
    # Send recognized text to OpenAI LLM
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": recognized_text}
            ]
        )
        answer = response.choices[0].message['content']
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    return jsonify({'recognized_text': recognized_text, 'response': answer})

if __name__ == '__main__':
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=True)
