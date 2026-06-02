import os
import tempfile
import uvicorn
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import av

# Initialize FastAPI App
app = FastAPI(
    title="Local Offline STT Server (Vosk Fallback)",
    description="Offline Speech-to-Text engine using Vosk running locally on CPU",
    version="1.0.0"
)

# Setup CORS to allow backend API forwarding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local model loaded lazily
model = None

def get_vosk_model():
    global model
    if model is None:
        try:
            import vosk
            print("Loading local offline STT model (Vosk en-us)...")
            # Automatically downloads the small English model if not present (~40MB)
            model = vosk.Model(lang="en-us")
            print("Model loaded successfully!")
        except ImportError:
            print("Error: 'vosk' package is not installed.")
            print("Please run: pip install -r requirements-stt.txt")
            raise HTTPException(
                status_code=500,
                detail="vosk is not installed. Run: pip install -r requirements-stt.txt"
            )
        except Exception as e:
            print(f"Failed to load Vosk model: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load local Vosk model: {str(e)}"
            )
    return model

def decode_audio_to_pcm(audio_path, target_sample_rate=16000):
    """
    Decodes an audio file of any format (WebM, OGG, WAV, etc.) to raw 16kHz mono PCM 16-bit bytes.
    Uses 'av' (PyAV) for robust in-memory decoding.
    """
    container = av.open(audio_path)
    
    # Find the first audio stream
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if audio_stream is None:
        raise ValueError("No audio stream found in the uploaded file.")
        
    resampler = av.AudioResampler(
        format='s16',
        layout='mono',
        rate=target_sample_rate
    )
    
    pcm_data = bytearray()
    
    for frame in container.decode(audio_stream):
        resampled_frames = resampler.resample(frame)
        for rf in resampled_frames:
            pcm_data.extend(rf.to_ndarray().tobytes())
            
    return bytes(pcm_data)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts audio file, decodes it to 16kHz PCM mono WAV bytes, and transcribes it locally using Vosk.
    """
    # 1. Get/Load model
    vosk_model = get_vosk_model()
    
    import vosk

    # 2. Write file to temp directory for processing
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"local_stt_{file.filename or 'audio.webm'}")
    
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 3. Decode WebM/OGG to raw PCM bytes
        pcm_bytes = decode_audio_to_pcm(temp_path, target_sample_rate=16000)
        
        # 4. Transcribe using Vosk KaldiRecognizer
        rec = vosk.KaldiRecognizer(vosk_model, 16000)
        rec.SetWords(False) # We don't need word-level timings
        
        # Feed data to recognizer
        chunk_size = 4000
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i:i+chunk_size]
            rec.AcceptWaveform(chunk)
            
        result_json = json.loads(rec.FinalResult())
        transcript_text = result_json.get("text", "").strip()
        
        print(f"[en-us] Local Vosk Transcription: {transcript_text}")
        
        return {
            "status": "success",
            "text": transcript_text,
            "language": "en"
        }
    except Exception as e:
        print(f"Local transcription error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Local transcription failed: {str(e)}"
        )
    finally:
        # Clean up temporary audio file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    print("Starting Local Vosk STT Server on http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)
