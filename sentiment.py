"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Sentiment Analysis & Virality Detection                   ║
║                  Emotional Intelligence for Video Content                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module analyzes the emotional intensity of video transcripts and
identifies potential viral moments using TextBlob sentiment analysis.
"""

import re
from typing import List, Tuple, Dict, Optional
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
import tempfile

# Configure matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')


def split_transcript_by_time(transcript: str, chunk_duration: int = 10) -> List[Tuple[int, str]]:
    """
    Split transcript into time-based chunks.
    
    Note: Since Whisper doesn't provide word-level timestamps by default,
    we'll estimate based on word count and average speaking rate.
    
    Args:
        transcript: Full transcript text
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        List of tuples (start_time_seconds, chunk_text)
    """
    # Average speaking rate: ~150 words per minute = 2.5 words per second
    words_per_second = 2.5
    words_per_chunk = int(chunk_duration * words_per_second)
    
    words = transcript.split()
    chunks = []
    
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i + words_per_chunk]
        chunk_text = ' '.join(chunk_words)
        start_time = int(i / words_per_second)
        chunks.append((start_time, chunk_text))
    
    return chunks


def analyze_sentiment(transcript: str, chunk_duration: int = 10) -> List[Dict]:
    """
    Analyze sentiment of transcript chunks.
    
    Args:
        transcript: Full transcript text
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        List of dictionaries with sentiment data:
        [
            {
                'start_time': 0,
                'end_time': 10,
                'text': 'chunk text',
                'polarity': 0.5,  # -1 (negative) to 1 (positive)
                'subjectivity': 0.6,  # 0 (objective) to 1 (subjective)
                'intensity': 0.55  # Combined metric
            },
            ...
        ]
    """
    if not transcript or transcript.strip() == "":
        return []
    
    chunks = split_transcript_by_time(transcript, chunk_duration)
    sentiment_data = []
    
    for start_time, chunk_text in chunks:
        if not chunk_text.strip():
            continue
            
        # Analyze sentiment using TextBlob
        blob = TextBlob(chunk_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Calculate intensity (combination of polarity magnitude and subjectivity)
        # High intensity = strong emotion (either positive or negative) + subjective
        intensity = (abs(polarity) + subjectivity) / 2
        
        sentiment_data.append({
            'start_time': start_time,
            'end_time': start_time + chunk_duration,
            'text': chunk_text[:100] + '...' if len(chunk_text) > 100 else chunk_text,
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'intensity': round(intensity, 3)
        })
    
    return sentiment_data


def detect_viral_clip(
    sentiment_data: List[Dict],
    window_duration: int = 45
) -> Optional[Dict]:
    """
    Detect the most intense 30-60 second window as potential viral clip.
    
    Args:
        sentiment_data: List of sentiment analysis results
        window_duration: Target duration for viral clip (seconds)
        
    Returns:
        Dictionary with viral clip info:
        {
            'start_time': 60,
            'end_time': 105,
            'avg_intensity': 0.75,
            'peak_emotion': 'positive',
            'description': 'High energy moment with strong positive sentiment'
        }
        or None if no suitable clip found
    """
    if not sentiment_data or len(sentiment_data) < 3:
        return None
    
    # Calculate how many chunks fit in the window
    chunk_duration = sentiment_data[0]['end_time'] - sentiment_data[0]['start_time']
    chunks_in_window = max(3, window_duration // chunk_duration)
    
    best_window = None
    max_avg_intensity = 0
    
    # Sliding window to find highest average intensity
    for i in range(len(sentiment_data) - chunks_in_window + 1):
        window = sentiment_data[i:i + chunks_in_window]
        avg_intensity = sum(chunk['intensity'] for chunk in window) / len(window)
        
        if avg_intensity > max_avg_intensity:
            max_avg_intensity = avg_intensity
            best_window = window
    
    if not best_window or max_avg_intensity < 0.3:  # Threshold for "interesting"
        return None
    
    # Determine peak emotion
    avg_polarity = sum(chunk['polarity'] for chunk in best_window) / len(best_window)
    
    if avg_polarity > 0.2:
        peak_emotion = 'positive'
        description = 'High energy moment with strong positive sentiment'
    elif avg_polarity < -0.2:
        peak_emotion = 'negative'
        description = 'Intense moment with strong negative/controversial sentiment'
    else:
        peak_emotion = 'neutral'
        description = 'Emotionally intense moment with mixed sentiment'
    
    return {
        'start_time': best_window[0]['start_time'],
        'end_time': best_window[-1]['end_time'],
        'avg_intensity': round(max_avg_intensity, 3),
        'peak_emotion': peak_emotion,
        'description': description
    }


def create_sentiment_graph(
    sentiment_data: List[Dict],
    viral_clip: Optional[Dict] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Create a matplotlib graph showing sentiment over time.
    
    Args:
        sentiment_data: List of sentiment analysis results
        viral_clip: Optional viral clip data to highlight
        save_path: Optional path to save the graph
        
    Returns:
        Path to the saved graph image
    """
    if not sentiment_data:
        # Create empty graph
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No sentiment data available',
                ha='center', va='center', fontsize=16, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    else:
        # Extract data for plotting
        times = [(d['start_time'] + d['end_time']) / 2 for d in sentiment_data]
        polarities = [d['polarity'] for d in sentiment_data]
        intensities = [d['intensity'] for d in sentiment_data]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Plot 1: Polarity (Positive/Negative)
        ax1.plot(times, polarities, color='#8b5cf6', linewidth=2.5, marker='o', 
                markersize=4, label='Sentiment Polarity')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.fill_between(times, polarities, 0, where=[p >= 0 for p in polarities],
                         color='#10b981', alpha=0.3, label='Positive')
        ax1.fill_between(times, polarities, 0, where=[p < 0 for p in polarities],
                         color='#ef4444', alpha=0.3, label='Negative')
        ax1.set_ylabel('Sentiment Polarity', fontsize=12, fontweight='bold')
        ax1.set_ylim(-1, 1)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Emotional Analysis Over Time', fontsize=14, fontweight='bold', pad=15)
        
        # Plot 2: Intensity
        ax2.plot(times, intensities, color='#fbbf24', linewidth=2.5, marker='s',
                markersize=4, label='Emotional Intensity')
        ax2.fill_between(times, intensities, 0, color='#fbbf24', alpha=0.3)
        ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Intensity', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Highlight viral clip if detected
        if viral_clip:
            for ax in [ax1, ax2]:
                ax.axvspan(viral_clip['start_time'], viral_clip['end_time'],
                          color='#ec4899', alpha=0.2, label='Viral Clip')
            
            # Add annotation
            mid_time = (viral_clip['start_time'] + viral_clip['end_time']) / 2
            ax1.annotate('Potential Viral Moment',
                        xy=(mid_time, 0.8),
                        xytext=(mid_time, 0.95),
                        arrowprops=dict(arrowstyle='->', color='#ec4899', lw=2),
                        fontsize=11, fontweight='bold', color='#ec4899',
                        ha='center')
    
    plt.tight_layout()
    
    # Save to file
    if save_path is None:
        save_path = tempfile.mktemp(suffix='.png', prefix='sentiment_')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path


def get_sentiment_summary(sentiment_data: List[Dict]) -> Dict:
    """
    Get overall sentiment statistics.
    
    Args:
        sentiment_data: List of sentiment analysis results
        
    Returns:
        Dictionary with summary statistics
    """
    if not sentiment_data:
        return {
            'avg_polarity': 0,
            'avg_intensity': 0,
            'overall_tone': 'neutral',
            'emotional_range': 0
        }
    
    polarities = [d['polarity'] for d in sentiment_data]
    intensities = [d['intensity'] for d in sentiment_data]
    
    avg_polarity = sum(polarities) / len(polarities)
    avg_intensity = sum(intensities) / len(intensities)
    emotional_range = max(polarities) - min(polarities)
    
    # Determine overall tone
    if avg_polarity > 0.2:
        overall_tone = 'positive'
    elif avg_polarity < -0.2:
        overall_tone = 'negative'
    else:
        overall_tone = 'neutral'
    
    return {
        'avg_polarity': round(avg_polarity, 3),
        'avg_intensity': round(avg_intensity, 3),
        'overall_tone': overall_tone,
        'emotional_range': round(emotional_range, 3)
    }


def format_timestamp(seconds: int) -> str:
    """
    Format seconds as MM:SS timestamp.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def slice_viral_clip(video_path: str, start_time: int, end_time: int) -> Optional[str]:
    """
    Extract the viral clip processing using moviepy.
    
    Args:
        video_path: Path to source video
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Path to extracted clip or None if error
    """
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        import os
        
        # Validate file
        if not os.path.exists(video_path):
            return None
            
        with VideoFileClip(video_path) as video:
            # Ensure times are within bounds
            if start_time < 0: start_time = 0
            if end_time > video.duration: end_time = video.duration
            if start_time >= end_time: return None
            
            # Extract clip
            clip = video.subclip(start_time, end_time)
            
            # Create output path
            output_path = tempfile.mktemp(suffix='.mp4', prefix='viral_clip_')
            
            # Write file (using fast settings for speed)
            clip.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac', 
                temp_audiofile='temp-audio.m4a', 
                remove_temp=True,
                logger=None  # Silence logger
            )
            
            return output_path
            
    except Exception as e:
        print(f"Error slicing clip: {str(e)}")
        return None
