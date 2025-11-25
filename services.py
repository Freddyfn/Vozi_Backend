"""
Vozi Backend Services

This module provides core services for the Vozi application:
- YouTube audio download (yt-dlp, Azure-compatible)
- Audio transcription (Gemini AI)
- Text analysis and correction (Gemini AI)

All services are designed to work seamlessly on Azure App Service without
requiring FFmpeg or other system dependencies.
"""

import os
import logging
import time
import tempfile
import shutil
from typing import Optional

import google.generativeai as genai
import yt_dlp
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Gemini AI
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    logger.info("Gemini API configured successfully")
else:
    logger.warning("GEMINI_API_KEY not found in environment variables")

# Gemini model configuration
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Supported audio MIME types
AUDIO_MIME_TYPES = {
    '.mp3': 'audio/mp3',
    '.wav': 'audio/wav',
    '.m4a': 'audio/m4a',
    '.aac': 'audio/aac',
    '.ogg': 'audio/ogg',
    '.opus': 'audio/opus',
    '.webm': 'audio/webm',
    '.flac': 'audio/flac',
}

# ============================================================================
# YOUTUBE DOWNLOAD SERVICE
# ============================================================================

async def download_youtube_audio(url: str) -> str:
    """
    Download audio from a YouTube URL in native format (Azure App Service compatible).
    
    This function downloads audio without FFmpeg conversion, making it suitable for
    deployment on Azure App Service where FFmpeg may not be available. Gemini AI
    can process various audio formats including m4a, webm, and opus.
    
    Args:
        url (str): YouTube URL to download audio from.
                   Supports youtube.com and youtu.be formats.
        
    Returns:
        str: Absolute path to the downloaded audio file in a temporary directory.
        
    Raises:
        Exception: If the download fails, URL is invalid, or file is not found.
        
    Example:
        >>> audio_path = await download_youtube_audio("https://youtube.com/watch?v=...")
        >>> print(f"Downloaded to: {audio_path}")
    
    Note:
        The caller is responsible for cleaning up the temporary directory
        after processing the audio file.
    """
    temp_dir = None
    try:
        # Create temporary directory for download
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, 'audio.%(ext)s')
        
        # Configure yt-dlp (optimized for Azure deployment)
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'quiet': True,  # Suppress verbose output
            'no_warnings': True,  # Suppress warnings
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios', 'android']  # Avoid JS runtime warnings
                }
            },
        }
        
        logger.info(f"Downloading audio from YouTube: {url}")

        # Download audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_title = info.get('title', 'Unknown')
            downloaded_file = ydl.prepare_filename(info)
            
        # Verify download
        if not os.path.exists(downloaded_file):
            raise Exception("Downloaded file not found after extraction")
            
        logger.info(f"Successfully downloaded: {video_title}")
        logger.info(f"Audio file saved: {downloaded_file}")
        
        return downloaded_file
                
    except Exception as e:
        logger.error(f"Failed to download YouTube audio: {str(e)}")
        
        # Cleanup on failure
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        raise Exception(f"Failed to download YouTube audio: {str(e)}")

# ============================================================================
# AUDIO TRANSCRIPTION SERVICE
# ============================================================================

def _detect_mime_type(file_path: str) -> str:
    """
    Auto-detect MIME type based on file extension.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        str: MIME type string (e.g., 'audio/mp3').
    """
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = AUDIO_MIME_TYPES.get(ext, 'audio/mp3')
    logger.info(f"Detected MIME type: {mime_type} for extension: {ext}")
    return mime_type

async def process_audio(file_path: str, mime_type: Optional[str] = None) -> str:
    """
    Upload and transcribe audio file using Gemini AI.
    
    This function handles the complete audio processing pipeline:
    1. Auto-detects MIME type if not provided
    2. Uploads audio to Gemini
    3. Waits for processing to complete
    4. Generates timestamped transcription
    5. Cleans up Gemini storage
    
    Args:
        file_path (str): Absolute path to the audio file.
        mime_type (str, optional): MIME type of the audio.
                                   If None, will be auto-detected from file extension.
        
    Returns:
        str: Transcribed text with timestamps in format [MM:SS] or [HH:MM:SS].
        
    Raises:
        Exception: If upload fails, processing fails, or Gemini returns an error.
        
    Example:
        >>> transcription = await process_audio("/tmp/audio.mp3")
        >>> print(transcription)
        [00:00] Welcome to the presentation.
        [00:15] Today we will discuss...
    """
    audio_file = None
    try:
        # Auto-detect MIME type if not provided
        if mime_type is None:
            mime_type = _detect_mime_type(file_path)
        
        # Upload to Gemini
        logger.info(f"Uploading audio file to Gemini: {file_path}")
        audio_file = genai.upload_file(file_path, mime_type=mime_type)
        logger.info(f"File uploaded successfully. URI: {audio_file.uri}")
        
        # Wait for processing
        while audio_file.state.name == "PROCESSING":
            logger.debug("Waiting for Gemini to process audio...")
            time.sleep(1)
            audio_file = genai.get_file(audio_file.name)
            
        if audio_file.state.name == "FAILED":
            raise Exception("Gemini audio processing failed")
            
        logger.info("Audio file ready for transcription")
        
        # Generate transcription
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = """
        You are an expert audio transcription AI with timestamp capabilities.
        
        **INSTRUCTIONS:**
        1.  **TRANSCRIPTION WITH TIMESTAMPS:** Transcribe the audio file EXACTLY as spoken, including timestamps.
        2.  **VERBATIM:** Do not correct grammar, spelling, or punctuation. Do not omit or add words.
        3.  **NO CENSORSHIP:** Transcribe ALL content faithfully, including explicit language, profanity, or offensive words. Do NOT censor, replace, or omit any words regardless of content.
        4.  **TIMESTAMP FORMAT:** Include timestamps in the format [MM:SS] or [HH:MM:SS] at natural breaks (sentences, phrases, or logical pauses).
        5.  **OUTPUT:** Return the transcribed text with timestamps. For example:
            [00:00] First sentence here.
            [00:15] Next sentence or phrase here.
        """
        
        response = model.generate_content([prompt, audio_file])
        logger.info("Transcription completed successfully")
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error in audio processing: {str(e)}")
        raise
        
    finally:
        # Clean up Gemini storage
        if audio_file:
            try:
                audio_file.delete()
                logger.debug("Cleaned up Gemini file storage")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup Gemini file: {cleanup_error}")

async def transcribe_audio(file_path: str) -> str:
    """
    Convenience wrapper for process_audio().
    
    Args:
        file_path (str): Path to the audio file to transcribe.
        
    Returns:
        str: Transcribed text with timestamps.
    """
    return await process_audio(file_path)

# ============================================================================
# TEXT ANALYSIS SERVICE
# ============================================================================

async def analyze_text(text: str) -> str:
    """
    Analyze and polish text using Gemini AI.
    
    This function:
    - Removes timestamps from transcriptions
    - Fixes spelling errors (typos only)
    - Corrects punctuation and capitalization
    - Formats text with markdown (bold, lists, headers)
    - Preserves original word choice (no paraphrasing)
    - Handles known works (songs, poems, etc.) with canonical text
    - Does NOT censor explicit content
    
    Args:
        text (str): Raw text to analyze (may include timestamps from transcription).
        
    Returns:
        str: Polished and formatted text in markdown format.
        
    Raises:
        Exception: If Gemini API fails or returns an error.
        
    Example:
        >>> raw = "[00:00] teh quick brown fox jumps over the lazy dog"
        >>> polished = await analyze_text(raw)
        >>> print(polished)
        The quick brown fox jumps over the lazy dog.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = """
        You are an expert professional editor and proofreader. Your goal is to polish the provided text ONLY by fixing errors, NOT by changing words.

        **CRITICAL RULE: DO NOT CHANGE WORDS**
        You are ONLY allowed to fix spelling errors, punctuation, and capitalization. You MUST NOT replace, substitute, or change any words to synonyms or "better" alternatives.

        **STRICT GUIDELINES:**

        1.  **CONTENT DETECTION (First Step):**
            *   Analyze if the text corresponds to a **known work**: famous literary piece, song lyrics, poem, speech, idiom, or quotation.
            *   **IF KNOWN WORK DETECTED:** Retrieve and use the **CANONICAL, EXACT ORIGINAL TEXT** from your knowledge base. Override any errors or variations in the input to match the official version. This includes explicit or profane content if it exists in the original.
            *   **IF GENERAL/UNKNOWN CONTENT:** Proceed to steps 2-7. DO NOT CHANGE ANY WORDS.

        2.  **NO CENSORSHIP:**
            *   Do NOT censor, replace, or omit explicit language, profanity, or offensive content.
            *   Preserve ALL words faithfully, regardless of their nature (songs, poems, or any content may contain explicit language that must be kept).

        3.  **REMOVE TIMESTAMPS:**
            *   If the input text contains timestamps (e.g., [00:00], [MM:SS], [HH:MM:SS]), **REMOVE THEM** from the output.
            *   Only keep the actual text content, without any time markers.

        4.  **SPELLING CORRECTION ONLY:**
            *   Fix spelling mistakes ONLY if they are clear typos (e.g., "teh" → "the").
            *   DO NOT change words to synonyms or "better" words.
            *   Keep the EXACT same words the user used.

        5.  **PUNCTUATION & CAPITALIZATION:**
            *   Fix punctuation to ensure proper sentence structure.
            *   Ensure proper capitalization (sentences, proper nouns).
            *   DO NOT change any words while doing this.

        6.  **ABSOLUTE WORD PRESERVATION:**
            *   **NEVER** paraphrase, rewrite, or summarize.
            *   **NEVER** change the user's choice of words, vocabulary, or tone.
            *   **NEVER** substitute words with synonyms or alternatives.
            *   **NEVER** add content that is not in the source text.
            *   The ONLY exception is fixing clear spelling typos.

        7.  **FORMATTING & STRUCTURE:**
            *   Organize the text into logical paragraphs.
            *   Use Markdown formatting to enhance readability:
                *   Use **bold** for emphasis if appropriate based on context.
                *   Use lists (bullet points) if the text lists items.
                *   Use headers (#, ##) only if there are clear section changes.

        8.  **FINAL OUTPUT:**
            *   Return **ONLY** the polished, formatted text with THE SAME WORDS.
            *   Do not include any introductory or concluding remarks.
        """
        
        logger.info("Analyzing text with Gemini AI")
        response = model.generate_content([prompt, text])
        logger.info("Text analysis completed successfully")
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        raise
