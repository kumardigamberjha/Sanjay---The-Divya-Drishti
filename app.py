"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Sanjaya v2.0 - The Content Command Center                 ║
║                          Divya Drishti • Divine Vision                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Production-Grade Video SEO & Strategy Tool                                   ║
║  100% Offline • Powered by Wings of AI                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARCHITECTURE:
=============
Phase A (EARS): Whisper → Audio Transcript
Phase B (EYES): OpenCV + LLaVA → Visual Analysis  
Phase C (BRAIN): Qwen2.5 → SEO Scoring + Recommendations

THE 7 PILLARS:
==============
1. System Health & Batch Queue
2. Sanjaya's Memory (Database)
3. Sentiment & Virality Detection
4. Competitor Benchmarking
5. Thumbnail A/B Scorer
6. Empire-Grade PDF Reports
7. Advanced Scoring Logic
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
from datetime import datetime
import pandas as pd
import altair as alt

# Import custom modules
import database
import sentiment
import pdf_gen
import utils

# ============================================================================
# CONFIGURATION
# ============================================================================
VISION_MODEL = "llava"
LOGIC_MODEL = "qwen2.5"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Sanjaya - The Divya Drishti",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING - Mystical Gold/Dark Theme
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0624 0%, #1a0b3d 50%, #0f0624 100%);
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Cinzel', serif !important;
        color: #fbbf24 !important;
        text-shadow: 0 0 20px rgba(251, 191, 36, 0.5);
    }
    
    /* Main header */
    .main-header {
        font-family: 'Cinzel', serif;
        background: linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        text-shadow: 0 0 30px rgba(251, 191, 36, 0.6);
    }
    
    .sub-header {
        color: #a78bfa;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26, 11, 61, 0.5);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #a78bfa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0b3d 0%, #0f0624 100%);
        border-right: 2px solid rgba(251, 191, 36, 0.3);
    }
    
    /* Metric boxes */
    .metric-box {
        background: linear-gradient(145deg, rgba(139, 92, 246, 0.2), rgba(124, 58, 237, 0.2));
        border: 2px solid rgba(251, 191, 36, 0.4);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
    }
    
    /* Status indicators */
    .status-online {
        color: #10b981;
        font-weight: bold;
    }
    
    .status-offline {
        color: #ef4444;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        color: #0f0624;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(251, 191, 36, 0.4);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(251, 191, 36, 0.6);
    }
    
    /* Text color */
    .stMarkdown, .stMarkdown p {
        color: #e0e7ff !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #fbbf24, transparent);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_whisper_model():
    """Load and cache Whisper model."""
    import whisper
    return whisper.load_model("base")


def check_system_health() -> dict:
    """
    Check health of all system components.
    
    Returns:
        Dictionary with status of each component
    """
    health = {
        'ollama': False,
        'llava': False,
        'qwen': False,
        'database': False
    }
    
    try:
        import ollama
        models_response = ollama.list()
        models_list = models_response.get('models', [])
        available_models = []
        
        for m in models_list:
            model_name = m.get('name') or m.get('model', '')
            if model_name:
                base_name = model_name.split(':')[0]
                available_models.append(base_name)
        
        health['ollama'] = True
        health['llava'] = VISION_MODEL in available_models
        health['qwen'] = LOGIC_MODEL in available_models
    except:
        pass
    
    try:
        database.init_database()
        health['database'] = True
    except:
        pass
    
    return health


def analyze_frame_with_llava(image: Image.Image, prompt: str) -> str:
    """Send frame to LLaVA for analysis."""
    try:
        import ollama
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_base64]
            }]
        )
        
        return response['message']['content']
    except Exception as e:
        return f"[Error: {str(e)}]"


def extract_audio_transcript(video_path: str) -> str:
    """Extract and transcribe audio using Whisper."""
    try:
        model = load_whisper_model()
        result = model.transcribe(video_path)
        transcript = result.get("text", "").strip()
        return transcript if transcript else "[No speech detected]"
    except Exception as e:
        return f"[Transcription error: {str(e)}]"


def extract_frames(video_path: str, interval_seconds: int = 5) -> list:
    """Extract frames from video."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return frames
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            frames.append((current_frame, pil_image))
        
        current_frame += 1
    
    cap.release()
    return frames


def get_seo_analysis(keyword: str, transcript: str, visual_summary: str, 
                     hook_analysis: str, keyword_density: float,
                     competitor_text: str = None) -> str:
    """Get SEO analysis from Qwen2.5."""
    try:
        import ollama
        
        # Extract first 5 seconds for hook analysis
        words = transcript.split()
        first_5_sec_words = words[:13]  # ~2.5 words/sec * 5 sec
        hook_text = ' '.join(first_5_sec_words)
        
        prompt = f"""You are Sanjaya, the divine strategist with the gift of Divya Drishti (divine vision).
Perform a STRICT and CRITICAL Video SEO analysis.

TARGET KEYWORD/NICHE: {keyword}

HOOK ANALYSIS (First 5 seconds):
{hook_text}
Hook Quality Assessment: {hook_analysis}

KEYWORD DENSITY: {keyword_density:.2f}%

FULL TRANSCRIPT (What was spoken):
{transcript[:3000]}

VISUAL SUMMARY (What was shown):
{visual_summary[:3000]}

{'COMPETITOR BENCHMARK:\n' + competitor_text[:1000] if competitor_text else ''}

YOUR DIVINE TASK:
Provide a strict, strategic analysis with:

1. **SEO RELEVANCE SCORE (0-100)**
   - Be harsh. Penalize weak hooks heavily (-15 points)
   - Penalize missing keyword in first 30 seconds (-10 points)
   - Only perfectly aligned videos score above 80

2. **HOOK QUALITY VERDICT**
   - Assess the opening 5 seconds critically
   
3. **KEYWORD PLACEMENT ANALYSIS**
   - Check if keyword appears early and naturally

4. **3 SPECIFIC ACTIONABLE RECOMMENDATIONS**
   - Be concrete and tactical
   - Reference specific moments from the content
   
{'5. **COMPETITOR COMPARISON**\n   - Highlight gaps vs benchmark\n   - Identify advantages' if competitor_text else ''}

Format in clean Markdown with headers."""

        response = ollama.chat(
            model=LOGIC_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']
    except Exception as e:
        return f"## ❌ Analysis Error\n\nSanjaya's vision is clouded: {str(e)}"


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔮 Sanjaya • Divya Drishti</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">The Content Command Center • Powered by Wings of AI</p>', unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR - System Health & Controls
    # ========================================================================
    with st.sidebar:
        st.markdown("## 🛡️ System Health")
        
        health = check_system_health()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Ollama:** {'🟢' if health['ollama'] else '🔴'}")
            st.markdown(f"**LLaVA:** {'🟢' if health['llava'] else '🔴'}")
        with col2:
            st.markdown(f"**Qwen:** {'🟢' if health['qwen'] else '🔴'}")
            st.markdown(f"**Database:** {'🟢' if health['database'] else '🔴'}")
        
        if not all([health['ollama'], health['llava'], health['qwen']]):
            st.error("⚠️ Some AI models are offline. Please start Ollama and pull required models.")
        
        st.divider()
        
        # Target Keyword
        st.markdown("### 🎯 Target Niche")
        target_keyword = st.text_input(
            "Enter keyword",
            placeholder="e.g., Python Tutorial",
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Competitor Benchmark (Pillar 4)
        st.markdown("### ⚔️ Competitor Spy Mode")
        enable_competitor = st.checkbox("Enable Benchmarking")
        competitor_text = None
        if enable_competitor:
            competitor_text = st.text_area(
                "Paste competitor transcript/description",
                height=100,
                placeholder="Enter competitor content to compare against..."
            )
        
        st.divider()
        
        # Thumbnail Scorer (Pillar 5)
        st.markdown("### 🖼️ Thumbnail A/B Tester")
        uploaded_thumbnails = st.file_uploader(
            "Upload thumbnail images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="thumbnails"
        )
    
    # ========================================================================
    # MAIN TABS
    # ========================================================================
    tab1, tab2, tab3 = st.tabs(["🎬 Analyze", "📚 Memory", "⚙️ Settings"])
    
    # ========================================================================
    # TAB 1: ANALYZE (Main Analysis Interface)
    # ========================================================================
    with tab1:
        st.markdown("### 📁 Upload Video for Divine Analysis")
        
        uploaded_file = st.file_uploader(
            "Select video file (.mp4, .mov)",
            type=['mp4', 'mov'],
            key="video"
        )
        
        if uploaded_file and target_keyword:
            # Show video metadata
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name
            
            metadata = utils.get_video_metadata(temp_video_path)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", utils.format_duration(metadata['duration']))
            with col2:
                st.metric("Resolution", metadata['resolution'])
            with col3:
                st.metric("FPS", f"{metadata['fps']}")
            with col4:
                st.metric("Size", utils.format_file_size(metadata['file_size_mb']))
            
            st.divider()
            
            # Analyze button
            if st.button("🔮 Channel Divine Insights", type="primary", use_container_width=True):
                try:
                    with st.status("🧘 Sanjaya is meditating on your content...", expanded=True) as status:
                        
                        # PHASE A: Audio Transcription
                        st.write("### 👂 Phase A: The Ears")
                        st.write("Whisper is transcribing the spoken words...")
                        transcript = extract_audio_transcript(temp_video_path)
                        st.write("✅ Transcript captured")
                        
                        # PHASE B: Visual Analysis
                        st.write("### 👁️ Phase B: The Eyes")
                        st.write("Extracting frames...")
                        frames = extract_frames(temp_video_path, interval_seconds=5)
                        st.write(f"Analyzing {len(frames)} frames with LLaVA...")
                        
                        visual_descriptions = []
                        progress_bar = st.progress(0)
                        for i, (frame_num, image) in enumerate(frames):
                            desc = analyze_frame_with_llava(
                                image,
                                "Describe this image briefly. List objects, text, and the setting."
                            )
                            visual_descriptions.append(f"Frame {i+1}: {desc}")
                            progress_bar.progress((i + 1) / len(frames))
                        
                        visual_summary = "\n".join(visual_descriptions)
                        st.write("✅ Visual analysis complete")
                        
                        # PILLAR 7: Advanced Scoring
                        st.write("### 🎯 Advanced Analysis")
                        
                        # Hook analysis
                        has_hook, hook_desc, hook_score = utils.analyze_hook(transcript, 5)
                        st.write(f"Hook Quality: {hook_score}/100")
                        
                        # Keyword density
                        keyword_density = utils.calculate_keyword_density(transcript, target_keyword)
                        st.write(f"Keyword Density: {keyword_density:.2f}%")
                        
                        # PILLAR 3: Sentiment Analysis
                        st.write("### 🎭 Sentiment Analysis")
                        sentiment_data = sentiment.analyze_sentiment(transcript, chunk_duration=10)
                        viral_clip = sentiment.detect_viral_clip(sentiment_data, window_duration=45)
                        sentiment_summary = sentiment.get_sentiment_summary(sentiment_data)
                        
                        # Create sentiment graph
                        graph_path = sentiment.create_sentiment_graph(sentiment_data, viral_clip)
                        st.write("✅ Emotional patterns analyzed")
                        
                        # PHASE C: SEO Analysis
                        st.write("### 🧠 Phase C: The Brain")
                        st.write("Qwen2.5 is formulating strategic insights...")
                        
                        seo_analysis = get_seo_analysis(
                            target_keyword, transcript, visual_summary,
                            hook_desc, keyword_density, competitor_text
                        )
                        
                        st.write("✅ Divine analysis complete!")
                        
                        status.update(label="✅ Sanjaya's Vision is Complete", state="complete", expanded=False)
                    
                    # Display Results
                    st.divider()
                    st.markdown("## 📊 Divine Revelations")
                    
                    # Sentiment Graph (Pillar 3)
                    st.markdown("### 🎭 Emotional Journey")
                    st.image(graph_path, use_container_width=True)
                    
                    if viral_clip:
                        st.success(f"🔥 **Potential Viral Clip Detected:** {sentiment.format_timestamp(viral_clip['start_time'])} - {sentiment.format_timestamp(viral_clip['end_time'])}")
                        st.caption(viral_clip['description'])
                    
                    # Two-column layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 👂 Audio Transcript")
                        with st.container(height=400):
                            st.markdown(transcript)
                    
                    with col2:
                        st.markdown("### 👁️ Visual Analysis")
                        with st.container(height=400):
                            st.markdown(visual_summary)
                    
                    # SEO Analysis
                    st.divider()
                    st.markdown("### 🧠 Strategic SEO Analysis")
                    st.markdown(seo_analysis)
                    
                    # Extract score from analysis (simple regex)
                    import re
                    score_match = re.search(r'(\d+)/100', seo_analysis)
                    seo_score = int(score_match.group(1)) if score_match else 50
                    
                    # PILLAR 2: Save to Database
                    viral_timestamp = f"{sentiment.format_timestamp(viral_clip['start_time'])}-{sentiment.format_timestamp(viral_clip['end_time'])}" if viral_clip else "None"
                    
                    record_id = database.save_analysis(
                        video_name=uploaded_file.name,
                        niche=target_keyword,
                        seo_score=seo_score,
                        viral_timestamp=viral_timestamp,
                        sentiment_avg=sentiment_summary['avg_polarity'],
                        hook_quality=hook_desc,
                        keyword_density=keyword_density,
                        transcript=transcript,
                        visual_summary=visual_summary,
                        recommendations=seo_analysis,
                        competitor_comparison=competitor_text if competitor_text else None
                    )
                    
                    st.success(f"✅ Analysis saved to Sanjaya's Memory (ID: {record_id})")
                    
                    # PILLAR 6: PDF Export
                    st.divider()
                    st.markdown("### 📄 Export Professional Report")
                    
                    if st.button("📥 Download Empire-Grade PDF", use_container_width=True):
                        pdf_bytes = pdf_gen.generate_report(
                            video_name=uploaded_file.name,
                            niche=target_keyword,
                            seo_score=seo_score,
                            viral_clip=viral_clip,
                            sentiment_summary=sentiment_summary,
                            hook_quality=hook_desc,
                            keyword_density=keyword_density,
                            recommendations=seo_analysis,
                            sentiment_graph_path=graph_path,
                            competitor_comparison=competitor_text
                        )
                        
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"sanjaya_report_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"🧘 Sanjaya's meditation was interrupted: {str(e)}")
                    st.exception(e)
                finally:
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
        
        elif uploaded_file:
            st.info("👆 Please enter a target keyword in the sidebar to begin analysis.")
        else:
            st.info("👆 Upload a video file to begin your journey with Sanjaya.")
        
        # PILLAR 5: Thumbnail Scoring
        if uploaded_thumbnails:
            st.divider()
            st.markdown("### 🖼️ Thumbnail A/B Test Results")
            
            for i, thumb_file in enumerate(uploaded_thumbnails, 1):
                with st.expander(f"Thumbnail {i}: {thumb_file.name}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        image = Image.open(thumb_file)
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        with st.spinner("LLaVA is analyzing..."):
                            score_result = analyze_frame_with_llava(
                                image,
                                """Rate this thumbnail 1-10 based on:
                                1. Clickability (visual appeal)
                                2. Text readability
                                3. Face emotion (if present)
                                Provide scores and brief explanation."""
                            )
                            st.markdown(score_result)
    
    # ========================================================================
    # TAB 2: MEMORY (History Dashboard)
    # ========================================================================
    with tab2:
        st.markdown("### 📚 Sanjaya's Memory")
        st.caption("All past analyses stored in the divine archive")
        
        # Get statistics
        stats = database.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Analyses", stats['total_analyses'])
        with col2:
            st.metric("Average Score", f"{stats['avg_score']:.1f}")
        with col3:
            st.metric("Highest Score", stats['max_score'])
        with col4:
            st.metric("Lowest Score", stats['min_score'])
        
        st.divider()
        
        # Get all analyses
        analyses = database.get_all_analyses(limit=50)
        
        if analyses:
            # Create DataFrame for display
            df_data = []
            for analysis in analyses:
                df_data.append({
                    'ID': analysis['id'],
                    'Video': analysis['video_name'],
                    'Niche': analysis['niche'],
                    'Score': analysis['seo_score'],
                    'Viral Clip': analysis['viral_timestamp'],
                    'Date': analysis['analysis_date']
                })
            
            df = pd.DataFrame(df_data)
            
            # Display table
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Score trend chart
            st.markdown("### 📈 Score Trend")
            trend_data = database.get_score_trend(limit=20)
            
            if trend_data:
                chart_df = pd.DataFrame(trend_data, columns=['Date', 'Score'])
                
                chart = alt.Chart(chart_df).mark_line(point=True).encode(
                    x='Date:T',
                    y='Score:Q',
                    tooltip=['Date', 'Score']
                ).properties(height=300)
                
                st.altair_chart(chart, use_container_width=True)
            
            # View detailed analysis
            st.divider()
            st.markdown("### 🔍 View Detailed Analysis")
            
            selected_id = st.selectbox(
                "Select analysis to view",
                options=[a['id'] for a in analyses],
                format_func=lambda x: f"ID {x}: {next(a['video_name'] for a in analyses if a['id'] == x)}"
            )
            
            if st.button("View Details"):
                analysis = database.get_analysis_by_id(selected_id)
                if analysis:
                    st.markdown(f"**Video:** {analysis['video_name']}")
                    st.markdown(f"**Niche:** {analysis['niche']}")
                    st.markdown(f"**Score:** {analysis['seo_score']}/100")
                    st.markdown(f"**Date:** {analysis['analysis_date']}")
                    
                    with st.expander("View Full Analysis"):
                        st.markdown(analysis['recommendations'])
        else:
            st.info("No analyses yet. Start by analyzing your first video!")
    
    # ========================================================================
    # TAB 3: SETTINGS
    # ========================================================================
    with tab3:
        st.markdown("### ⚙️ System Configuration")
        
        st.markdown("#### 🤖 AI Models")
        st.code(f"Vision Model: {VISION_MODEL}")
        st.code(f"Logic Model: {LOGIC_MODEL}")
        st.code(f"Audio Model: whisper (base)")
        
        st.divider()
        
        st.markdown("#### 🗄️ Database Management")
        st.caption(f"Database location: {database.DB_PATH}")
        
        if st.button("🗑️ Clear All History", type="secondary"):
            if st.checkbox("I understand this will delete all analyses"):
                count = database.clear_all_analyses()
                st.success(f"Cleared {count} analyses from memory")
        
        st.divider()
        
        st.markdown("#### 📊 About Sanjaya")
        st.markdown("""
        **Sanjaya • Divya Drishti** is a production-grade Video SEO & Strategy tool 
        powered by Wings of AI. Named after the divine narrator of the Mahabharata 
        who possessed the gift of seeing everything, Sanjaya brings divine vision 
        to your content strategy.
        
        **The 7 Pillars:**
        1. 🛡️ System Health & Batch Queue
        2. 🧠 Sanjaya's Memory (Database)
        3. 📈 Sentiment & Virality Detection
        4. ⚔️ Competitor Benchmarking
        5. 🖼️ Thumbnail A/B Scorer
        6. 📄 Empire-Grade PDF Reports
        7. 🎯 Advanced Scoring Logic
        
        **Version:** 2.0  
        **Powered by:** Wings of AI  
        **Models:** Ollama (LLaVA + Qwen2.5) + Whisper
        """)


if __name__ == "__main__":
    main()
