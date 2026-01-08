"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          Utility Functions                                    ║
║                Helper Functions for Video Analysis                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module provides shared utility functions for video metadata extraction,
keyword analysis, and hook quality assessment.
"""

import cv2
from typing import Dict, Tuple
import re


def get_video_metadata(video_path: str) -> Dict:
    """
    Extract metadata from video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video metadata:
        {
            'duration': 154.5,  # seconds
            'fps': 30.0,
            'resolution': '1920x1080',
            'total_frames': 4635,
            'file_size_mb': 45.2
        }
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {
            'duration': 0,
            'fps': 0,
            'resolution': 'Unknown',
            'total_frames': 0,
            'file_size_mb': 0
        }
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    # Get file size
    try:
        import os
        file_size_bytes = os.path.getsize(video_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
    except:
        file_size_mb = 0
    
    return {
        'duration': round(duration, 1),
        'fps': round(fps, 1),
        'resolution': f'{width}x{height}',
        'total_frames': total_frames,
        'file_size_mb': round(file_size_mb, 1)
    }


def calculate_keyword_density(transcript: str, keyword: str) -> float:
    """
    Calculate keyword density in transcript.
    
    Args:
        transcript: Full transcript text
        keyword: Target keyword/phrase
        
    Returns:
        Keyword density as percentage (0-100)
    """
    if not transcript or not keyword:
        return 0.0
    
    # Normalize text
    transcript_lower = transcript.lower()
    keyword_lower = keyword.lower()
    
    # Count total words
    words = transcript_lower.split()
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Count keyword occurrences (handle multi-word keywords)
    keyword_words = keyword_lower.split()
    keyword_count = 0
    
    if len(keyword_words) == 1:
        # Single word keyword
        keyword_count = words.count(keyword_lower)
    else:
        # Multi-word keyword - search for phrase
        keyword_count = transcript_lower.count(keyword_lower)
    
    # Calculate density: (keyword_count / total_words) * 100
    density = (keyword_count / total_words) * 100
    
    
    return round(density, 2)


def analyze_detailed_hook(
    video_path: str,
    transcript_segments: list,
    first_frame_analysis: str
) -> dict:
    """
    Analyze hook quality (first 5 seconds) based on audio and visual cues.
    
    Args:
        video_path: Path to video file
        transcript_segments: List of whisper segments (or text if segments not available)
        first_frame_analysis: LLaVA description of the first frame
        
    Returns:
        Dictionary with hook metrics and score
    """
    score = 10
    penalties = []
    strengths = []
    
    # 1. Audio Check (Speech in first 1.5s)
    # Note: If transcript_segments is just text, we can't fully check timing
    speech_start_time = 0
    has_early_speech = False
    
    if isinstance(transcript_segments, list) and len(transcript_segments) > 0:
        first_segment = transcript_segments[0]
        # Whisper segment structure: {'start': 0.0, 'end': 2.0, 'text': '...'}
        if isinstance(first_segment, dict):
            speech_start_time = first_segment.get('start', 0)
            if speech_start_time <= 1.5:
                has_early_speech = True
                strengths.append("Immediate speech detected (< 1.5s)")
            else:
                penalties.append("Delayed speech (start > 1.5s)")
                score -= 3
    else:
        # Fallback if no segments provided (assume speech exists if text exists)
        has_early_speech = True
        strengths.append("Speech detected (timing estimated)")

    # 2. Visual Check (Face or Text in first frame)
    first_frame_lower = first_frame_analysis.lower()
    has_face = any(x in first_frame_lower for x in ['face', 'person', 'man', 'woman', 'human'])
    has_text = any(x in first_frame_lower for x in ['text', 'words', 'title', 'caption', 'letters'])
    
    if has_face:
        strengths.append("Human presence in opening frame")
    if has_text:
        strengths.append("Text/Title visible in opening frame")
        
    if not has_face and not has_text:
        penalties.append("No clear face or text in opening frame")
        score -= 2
        
    # 3. Overall Verdict
    verdict = "Excellent Hook"
    if score < 5:
        verdict = "Weak Hook"
    elif score < 8:
        verdict = "Average Hook"
        
    return {
        'score': max(0, score),
        'verdict': verdict,
        'has_early_speech': has_early_speech,
        'has_face': has_face,
        'has_text': has_text,
        'strengths': strengths,
        'penalties': penalties
    }


def analyze_hook(transcript: str, duration_seconds: int = 5) -> Tuple[bool, str, int]:
    """
    Analyze the quality of the video hook (first few seconds).
    
    Args:
        transcript: Full transcript text
        duration_seconds: Duration of hook to analyze
        
    Returns:
        Tuple of (has_strong_hook, quality_description, quality_score)
        quality_score: 0-100
    """
    if not transcript or transcript.strip() == "":
        return False, "No speech detected in the opening. Silent intros reduce engagement.", 0
    
    # Estimate words in first N seconds (assuming ~2.5 words per second)
    words_per_second = 2.5
    expected_words = int(duration_seconds * words_per_second)
    
    words = transcript.split()
    hook_words = words[:expected_words]
    hook_text = ' '.join(hook_words).lower()
    
    # Initialize score
    score = 50  # Base score
    quality_notes = []
    
    # Check 1: Sufficient content
    if len(hook_words) < 5:
        score -= 30
        quality_notes.append("Hook is too short or sparse")
    elif len(hook_words) >= 10:
        score += 10
        quality_notes.append("Good content density")
    
    # Check 2: Starts with a question (engaging)
    question_starters = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'can', 'do', 'does', 'is', 'are']
    if any(hook_text.startswith(q) for q in question_starters) or '?' in hook_text:
        score += 15
        quality_notes.append("Opens with a question (engaging)")
    
    # Check 3: Contains power words
    power_words = ['discover', 'secret', 'revealed', 'shocking', 'amazing', 'ultimate', 
                   'proven', 'guaranteed', 'free', 'new', 'exclusive', 'limited']
    if any(word in hook_text for word in power_words):
        score += 10
        quality_notes.append("Uses power words")
    
    # Check 4: Contains numbers (specific and credible)
    if re.search(r'\d+', hook_text):
        score += 5
        quality_notes.append("Includes specific numbers")
    
    # Check 5: Avoid filler words
    filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually']
    filler_count = sum(hook_text.count(filler) for filler in filler_words)
    if filler_count > 2:
        score -= 10
        quality_notes.append("Contains excessive filler words")
    
    # Cap score at 0-100
    score = max(0, min(100, score))
    
    # Determine overall assessment
    if score >= 70:
        has_strong_hook = True
        description = "Strong hook! " + " ".join(quality_notes)
    elif score >= 50:
        has_strong_hook = True
        description = "Decent hook. " + " ".join(quality_notes)
    else:
        has_strong_hook = False
        description = "Weak hook. " + " ".join(quality_notes)
    
    return has_strong_hook, description, score


def check_keyword_in_first_n_seconds(
    transcript: str,
    keyword: str,
    duration_seconds: int = 30
) -> Tuple[bool, int]:
    """
    Check if keyword appears in the first N seconds of transcript.
    
    Args:
        transcript: Full transcript text
        keyword: Target keyword
        duration_seconds: Duration to check
        
    Returns:
        Tuple of (keyword_found, occurrence_count)
    """
    if not transcript or not keyword:
        return False, 0
    
    # Estimate words in first N seconds
    words_per_second = 2.5
    expected_words = int(duration_seconds * words_per_second)
    
    words = transcript.split()
    opening_text = ' '.join(words[:expected_words]).lower()
    keyword_lower = keyword.lower()
    
    # Count occurrences
    if ' ' in keyword_lower:
        # Multi-word keyword
        count = opening_text.count(keyword_lower)
    else:
        # Single word
        count = opening_text.split().count(keyword_lower)
    
    return count > 0, count


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2:34" or "1:05:23")
    """
    seconds = int(seconds)
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def format_file_size(size_mb: float) -> str:
    """
    Format file size for display.
    
    Args:
        size_mb: Size in megabytes
        
    Returns:
        Formatted string (e.g., "45.2 MB" or "1.2 GB")
    """
    if size_mb >= 1024:
        size_gb = size_mb / 1024
        return f"{size_gb:.1f} GB"
    else:
        return f"{size_mb:.1f} MB"


def extract_recommendations_list(recommendations_text: str) -> list:
    """
    Extract individual recommendations from formatted text.
    
    Args:
        recommendations_text: Formatted recommendations string
        
    Returns:
        List of individual recommendation strings
    """
    # Split by common delimiters
    lines = recommendations_text.split('\n')
    
    recommendations = []
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
            # Remove numbering/bullets
            clean_line = re.sub(r'^[\d\-•\.\)]+\s*', '', line)
            if clean_line:
                recommendations.append(clean_line)
    
    return recommendations[:3]  # Return top 3
