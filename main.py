"""
Vozi Backend API

FastAPI application providing endpoints for:
- Audio file transcription
- YouTube video audio transcription  
- Text analysis and correction

Designed for deployment on Azure App Service with React frontend.
"""

import os
import logging
import re
import shutil
import tempfile
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services import transcribe_audio, analyze_text, download_youtube_audio

# ============================================================================
# CONFIGURATION
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vozi API",
    description="Audio transcription and text analysis API powered by Gemini AI",
    version="1.0.0"
)

# CORS configuration for frontend
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YouTube URL validation regex
YOUTUBE_URL_REGEX = r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$'

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/api/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Transcribe an uploaded audio file.
    
    Accepts audio files in various formats (mp3, wav, m4a, webm, etc.),
    saves temporarily, transcribes using Gemini AI, and returns the result.
    
    Args:
        file (UploadFile): Audio file from multipart/form-data request.
                          Supports: mp3, wav, m4a, aac, ogg, opus, webm, flac
    
    Returns:
        dict: JSON response containing the transcription.
              Example: {"transcription": "Transcribed text with [00:00] timestamps..."}
    
    Raises:
        HTTPException: 500 if transcription fails.
        
    Example Request:
        POST /api/transcribe
        Content-Type: multipart/form-data
        Body: file=@audio.mp3
    """
    temp_file_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=os.path.splitext(file.filename)[1]
        ) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Transcribe audio
        transcription = await transcribe_audio(temp_file_path)
        
        logger.info("Transcription completed successfully")
        return {"transcription": transcription}
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Cleaned up temporary file: {temp_file_path}")

@app.post("/api/analyze")
async def analyze_endpoint(data: Dict) -> Dict[str, str]:
    """
    Analyze and polish text.
    
    Takes raw text (potentially from transcription) and:
    - Removes timestamps
    - Fixes spelling errors (typos only)
    - Corrects punctuation and capitalization
    - Formats with markdown
    - Preserves original word choice
    
    Args:
        data (dict): JSON body with "text" field.
                     Example: {"text": "[00:00] teh quick brown fox..."}
    
    Returns:
        dict: JSON response containing the polished text.
              Example: {"analysis": "The quick brown fox..."}
    
    Raises:
        HTTPException: 400 if "text" field is missing.
        HTTPException: 500 if analysis fails.
        
    Example Request:
        POST /api/analyze
        Content-Type: application/json
        Body: {"text": "Raw transcription text..."}
    """
    text = data.get("text")
    
    if not text:
        raise HTTPException(
            status_code=400, 
            detail="'text' field is required in request body"
        )
    
    try:
        logger.info("Processing text analysis request")
        analysis = await analyze_text(text)
        
        logger.info("Text analysis completed successfully")
        return {"analysis": analysis}
        
    except Exception as e:
        logger.error(f"Text analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe-youtube")
async def transcribe_youtube_endpoint(data: Dict) -> Dict[str, str]:
    """
    Transcribe audio from a YouTube URL using hybrid approach.
    
    This endpoint uses a two-step fallback strategy:
    1. Try to download audio with yt-dlp (with anti-bot headers)
    2. If that fails, try to get transcript directly from YouTube (for videos with captions)
    
    Args:
        data (dict): JSON body with "url" field.
                     Example: {"url": "https://youtube.com/watch?v=..."}
                     Supported: youtube.com and youtu.be URLs
    
    Returns:
        dict: JSON response containing the transcription.
              Example: {"transcription": "Transcribed text with timestamps..."}
    
    Raises:
        HTTPException: 400 if "url" field is missing or invalid.
        HTTPException: 500 if both download and transcript fallback fail.
        
    Example Request:
        POST /api/transcribe-youtube
        Content-Type: application/json
        Body: {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    """
    url = data.get("url")
    
    if not url:
        raise HTTPException(
            status_code=400, 
            detail="'url' field is required in request body"
        )
    
    # Validate YouTube URL format
    if not re.match(YOUTUBE_URL_REGEX, url):
        raise HTTPException(
            status_code=400, 
            detail="Invalid YouTube URL. Supported formats: youtube.com, youtu.be"
        )
    
    audio_file_path = None
    try:
        # Download audio from YouTube (with fallback to transcript)
        logger.info(f"Processing YouTube URL: {url}")
        audio_file_path = await download_youtube_audio(url)
        
        # Check if we got a transcript fallback (marked with TRANSCRIPT: prefix)
        if audio_file_path.startswith("TRANSCRIPT:"):
            # Read the transcript file directly
            transcript_file = audio_file_path.replace("TRANSCRIPT:", "")
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcription = f.read()
            
            logger.info("YouTube transcript retrieved successfully (fallback method)")
            return {"transcription": transcription}
        
        # We have an actual audio file, transcribe it with Gemini
        transcription = await transcribe_audio(audio_file_path)
        
        logger.info("YouTube audio transcription completed successfully")
        return {"transcription": transcription}
        
    except Exception as e:
        logger.error(f"YouTube transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary directory
        if audio_file_path:
            # Handle both regular files and transcript markers
            file_path = audio_file_path.replace("TRANSCRIPT:", "") if audio_file_path.startswith("TRANSCRIPT:") else audio_file_path
            
            if os.path.exists(file_path):
                temp_dir = os.path.dirname(file_path)
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")

# ============================================================================
# STATIC FILE SERVING (Production)
# ============================================================================

# Serve React frontend assets
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """
    Serve React SPA for all routes (catch-all for client-side routing).
    
    This endpoint ensures that React Router works correctly by serving
    index.html for all non-API routes.
    """
    return FileResponse("static/index.html")
