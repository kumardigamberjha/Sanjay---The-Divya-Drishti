"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              🎬 OptiView                                      ║
║                       Local Video SEO Auditor                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  100% Offline Video Analysis using Local AI Models                           ║
║                                                                              ║
║  Pipeline: Video → [Whisper: Audio] → [LLaVA: Vision] → [Qwen: Analysis]     ║
╚══════════════════════════════════════════════════════════════════════════════╝

DATA FLOW ARCHITECTURE:
=======================
1. VIDEO INPUT → User uploads .mp4/.mov file
2. PHASE A (EARS) → Whisper extracts and transcribes audio → transcript
3. PHASE B (EYES) → OpenCV extracts frames → LLaVA describes visuals → visual_summary
4. PHASE C (BRAIN) → Qwen2.5 analyzes transcript + visual_summary → SEO Score + Tips

"""

import streamlit as st
import cv2
import tempfile
import os
import base64
from pathlib import Path
import numpy as np
from PIL import Image
import io

# ============================================================================
# CONFIGURATION - Model Names for Ollama
# ============================================================================
VISION_MODEL = "llava"      # For analyzing video frames
LOGIC_MODEL = "qwen2.5"     # For reasoning and SEO grading

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Sanjaya - The Divya Drishti",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
<style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #ffffff 100%);
        color: #0f172a;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }

    .sub-header {
        color: #475569;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Card styling */
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
        color: #0f172a;
    }

    /* Sidebar */
    .css-1d391kg {
        background: #f1f5f9;
    }

    /* Success box */
    .success-box {
        background: #ecfdf5;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #22c55e;
        color: #065f46;
    }

    /* Info box */
    .info-box {
        background: #eff6ff;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #3b82f6;
        color: #1e3a8a;
    }

    /* Progress text */
    .progress-text {
        color: #64748b;
        font-style: italic;
    }

    /* Score */
    .score-display {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .score-label {
        color: #64748b;
        text-align: center;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_ollama_connection():
    """
    Verify that Ollama is running and required models are available.
    Returns: (bool, str) - (success, message)
    """
    try:
        import ollama
        # Try to list models to check connection
        models_response = ollama.list()
        
        # Handle different API response formats (older and newer Ollama versions)
        models_list = models_response.get('models', [])
        available_models = []
        
        for m in models_list:
            # Try both 'name' and 'model' keys for compatibility
            model_name = m.get('name') or m.get('model', '')
            if model_name:
                # Extract base model name (before the colon)
                base_name = model_name.split(':')[0]
                available_models.append(base_name)
                available_models.append(model_name)  # Also keep full name
        
        missing = []
        if VISION_MODEL not in available_models:
            missing.append(VISION_MODEL)
        if LOGIC_MODEL not in available_models:
            missing.append(LOGIC_MODEL)
        
        if missing:
            return False, f"Missing models: {', '.join(missing)}. Run: `ollama pull {' && ollama pull '.join(missing)}`"
        
        return True, "Ollama is running with all required models."
    except ImportError:
        return False, "Ollama Python library not installed. Run: `pip install ollama`"
    except Exception as e:
        return False, f"Cannot connect to Ollama. Make sure it's running (`ollama serve`). Error: {str(e)}"


def extract_audio_transcript(video_path: str, progress_callback=None) -> str:
    """
    PHASE A: THE EARS
    ==================
    Extract audio from video and transcribe using Whisper.
    
    Args:
        video_path: Path to the video file
        progress_callback: Optional function to update UI progress
        
    Returns:
        Full text transcript of the audio
    """
    try:
        import whisper
        
        if progress_callback:
            progress_callback("Loading Whisper model (base)...")
        
        # Load the base Whisper model (good balance of speed/accuracy)
        model = whisper.load_model("base")
        
        if progress_callback:
            progress_callback("Transcribing audio... This may take a moment.")
        
        # Transcribe the audio track from the video
        result = model.transcribe(video_path)
        
        transcript = result.get("text", "").strip()
        
        if not transcript:
            return "[No speech detected in the video]"
        
        return transcript
        
    except Exception as e:
        return f"[Audio transcription error: {str(e)}]"


def extract_frames(video_path: str, interval_seconds: int = 5) -> list:
    """
    Extract frames from video at specified intervals.
    
    Args:
        video_path: Path to the video file
        interval_seconds: Extract one frame every N seconds
        
    Returns:
        List of (frame_number, PIL Image) tuples
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval_seconds)
    
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            # Resize for efficiency (max 512px on longest side)
            pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            frames.append((current_frame, pil_image))
        
        current_frame += 1
    
    cap.release()
    return frames


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for Ollama."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def analyze_frame_with_llava(image: Image.Image, frame_num: int) -> str:
    """
    PHASE B: THE EYES (Single Frame)
    ==================================
    Send a frame to LLaVA for visual description.
    
    Args:
        image: PIL Image of the frame
        frame_num: Frame number for reference
        
    Returns:
        Text description of the frame
    """
    try:
        import ollama
        
        # Convert image to base64
        img_base64 = image_to_base64(image)
        
        # Send to LLaVA with specific prompt
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': 'Describe this image briefly. List objects, text, and the setting.',
                'images': [img_base64]
            }]
        )
        
        return response['message']['content']
        
    except Exception as e:
        return f"[Frame analysis error: {str(e)}]"


def analyze_all_frames(video_path: str, status_container) -> str:
    """
    PHASE B: THE EYES (Full Video)
    ================================
    Extract frames and analyze each with LLaVA.
    
    Args:
        video_path: Path to the video file
        status_container: Streamlit container for progress updates
        
    Returns:
        Combined visual summary of all analyzed frames
    """
    # Extract frames (1 every 5 seconds)
    status_container.write("📹 Extracting frames from video...")
    frames = extract_frames(video_path, interval_seconds=5)
    
    if not frames:
        return "[No frames could be extracted from the video]"
    
    status_container.write(f"🖼️ Extracted {len(frames)} frames. Analyzing with LLaVA...")
    
    descriptions = []
    progress_bar = status_container.progress(0)
    progress_text = status_container.empty()
    
    for i, (frame_num, image) in enumerate(frames):
        progress_text.write(f"🔍 Analyzing frame {i+1}/{len(frames)}...")
        
        description = analyze_frame_with_llava(image, frame_num)
        
        # Format: Frame X at Y seconds: description
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        timestamp = frame_num / fps if fps > 0 else 0
        descriptions.append(f"**Frame at {timestamp:.1f}s:** {description}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(frames))
    
    progress_text.write("✅ Visual analysis complete!")
    
    return "\n\n".join(descriptions)


def get_seo_analysis(keyword: str, transcript: str, visual_summary: str) -> str:
    """
    PHASE C: THE BRAIN
    ===================
    Use Qwen2.5 to analyze the video content against the target keyword
    and provide SEO recommendations.
    
    Args:
        keyword: Target keyword/niche from user
        transcript: Audio transcript from Phase A
        visual_summary: Visual descriptions from Phase B
        
    Returns:
        Markdown-formatted SEO analysis with score and tips
    """
    try:
        import ollama
        
        # Construct the analysis prompt for Qwen2.5
        prompt = f"""You are a strict Video SEO Expert. Your job is to critically analyze video content.

## TARGET KEYWORD/NICHE:
{keyword}

## AUDIO TRANSCRIPT (What was said):
{transcript[:3000]}  # Limit to prevent context overflow

## VISUAL SUMMARY (What was shown):
{visual_summary[:3000]}  # Limit to prevent context overflow

## YOUR TASK:

Perform a STRICT and CRITICAL analysis of how well this video's audio and visuals align with the target keyword/niche.

You MUST provide:

1. **RELEVANCE SCORE (0-100)**: 
   - Be harsh. Only videos that are perfectly aligned should score above 80.
   - Average videos should score 40-60.
   - Videos with weak keyword alignment should score below 40.

2. **SCORE JUSTIFICATION**: 
   - Explain exactly why you gave this score.
   - Point out what's missing or wrong.

3. **3 SPECIFIC ACTIONABLE TIPS**:
   - These must be concrete, implementable suggestions.
   - Don't be generic. Reference specific elements from the transcript/visuals.
   - Focus on what would actually improve SEO performance.

Format your response in clean Markdown with headers and bullet points."""

        response = ollama.chat(
            model=LOGIC_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt
            }]
        )
        
        return response['message']['content']
        
    except Exception as e:
        return f"## ❌ Analysis Error\n\nCould not complete SEO analysis: {str(e)}"


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Sanjaya - The Divya Drishti</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Video Auditor • 100% Offline • Powered by Coding India</p>', unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR - Inputs
    # ========================================================================
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Check Ollama status
        ollama_ok, ollama_msg = check_ollama_connection()
        if ollama_ok:
            st.success(f"✅ {ollama_msg}")
        else:
            st.error(f"❌ {ollama_msg}")
            st.stop()
        
        st.divider()
        
        # Target Keyword Input
        st.markdown("### 🎯 Target Keyword/Niche")
        target_keyword = st.text_input(
            "Enter keyword",
            placeholder="e.g., Python Tutorial, Cooking Recipe, Gaming Review",
            help="The main topic or niche your video is targeting for SEO"
        )
        
        st.divider()
        
        # Video Upload
        st.markdown("### 📁 Upload Video")
        uploaded_file = st.file_uploader(
            "Select video file",
            type=['mp4', 'mov'],
            help="Upload a .mp4 or .mov video file for analysis"
        )
        
        st.divider()
        
        # Model Info
        with st.expander("🤖 Model Information"):
            st.markdown(f"""
            **Vision Model:** `{VISION_MODEL}`  
            *Analyzes video frames*
            
            **Logic Model:** `{LOGIC_MODEL}`  
            *Performs SEO analysis*
            
            **Audio Model:** `whisper (base)`  
            *Transcribes speech*
            """)
        
        st.divider()
        
        # Analyze Button
        analyze_button = st.button(
            "🚀 Analyze Video",
            type="primary",
            use_container_width=True,
            disabled=not (uploaded_file and target_keyword)
        )
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    if not uploaded_file:
        # Welcome screen
        st.markdown("""
        <div class="metric-card">
            <h3>👋 Welcome to OptiView!</h3>
            <p>This tool analyzes your videos for SEO optimization using 100% local AI models.</p>
            <br>
            <h4>📋 How it works:</h4>
            <ol>
                <li><strong>Upload</strong> - Select a .mp4 or .mov video file</li>
                <li><strong>Set Keyword</strong> - Enter your target SEO keyword/niche</li>
                <li><strong>Analyze</strong> - Click the button and wait for AI analysis</li>
                <li><strong>Review</strong> - Get your SEO score and actionable tips</li>
            </ol>
            <br>
            <h4>🔧 Pipeline:</h4>
            <p><code>Video → 👂 Whisper (Audio) → 👁️ LLaVA (Vision) → 🧠 Qwen (Analysis)</code></p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if not target_keyword:
        st.warning("⚠️ Please enter a target keyword/niche in the sidebar to continue.")
        return
    
    # If analyze button is clicked, run the full pipeline
    if analyze_button:
        # Save uploaded file to temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
        
        try:
            # Main processing status container
            with st.status("🎬 Analyzing video...", expanded=True) as status:
                
                # ============================================================
                # PHASE A: THE EARS - Audio Transcription
                # ============================================================
                st.write("### 👂 Phase A: Audio Analysis")
                st.write("Loading Whisper model and transcribing audio...")
                
                transcript = extract_audio_transcript(
                    temp_video_path,
                    progress_callback=lambda msg: st.write(f"   {msg}")
                )
                
                st.write("✅ Audio transcription complete!")
                
                # ============================================================
                # PHASE B: THE EYES - Visual Analysis
                # ============================================================
                st.write("### 👁️ Phase B: Visual Analysis")
                
                visual_summary = analyze_all_frames(temp_video_path, st)
                
                st.write("✅ Visual analysis complete!")
                
                # ============================================================
                # PHASE C: THE BRAIN - SEO Analysis
                # ============================================================
                st.write("### 🧠 Phase C: SEO Analysis")
                st.write(f"Analyzing content against keyword: **{target_keyword}**")
                
                seo_analysis = get_seo_analysis(target_keyword, transcript, visual_summary)
                
                st.write("✅ SEO analysis complete!")
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
            
            # ==================================================================
            # DISPLAY RESULTS
            # ==================================================================
            
            st.divider()
            st.markdown("## 📊 Analysis Results")
            
            # Two-column layout for transcript and visual analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 👂 Audio Transcript")
                with st.container(height=400):
                    st.markdown(transcript)
            
            with col2:
                st.markdown("### 👁️ Visual Analysis")
                with st.container(height=400):
                    st.markdown(visual_summary)
            
            # SEO Analysis (full width)
            st.divider()
            st.markdown("### 🧠 SEO Analysis & Recommendations")
            st.markdown(f"> **Target Keyword:** `{target_keyword}`")
            st.markdown(seo_analysis)
            
            # Success message
            st.balloons()
            st.success("🎉 Analysis complete! Review the results above for SEO recommendations.")
            
        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            st.exception(e)
        
        finally:
            # Cleanup temp file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    
    else:
        # Show video preview
        st.markdown("### 📹 Video Preview")
        st.video(uploaded_file)
        st.info("👆 Video loaded. Click **'🚀 Analyze Video'** in the sidebar to start the analysis.")


if __name__ == "__main__":
    main()
