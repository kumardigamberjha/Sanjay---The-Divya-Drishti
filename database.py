"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         Sanjaya's Memory System                               ║
║                    SQLite Database for Analysis History                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module handles all database operations for storing and retrieving
video analysis history, enabling the "Memory" feature of Sanjaya.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Database file path
DB_PATH = Path(__file__).parent / "sanjaya.db"


def init_database() -> sqlite3.Connection:
    """
    Initialize the database and create tables if they don't exist.
    
    Returns:
        sqlite3.Connection: Database connection object
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    cursor = conn.cursor()
    
    # Create analyses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT NOT NULL,
            niche TEXT NOT NULL,
            seo_score INTEGER,
            viral_timestamp TEXT,
            sentiment_avg REAL,
            hook_quality TEXT,
            keyword_density REAL,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            transcript TEXT,
            visual_summary TEXT,
            recommendations TEXT,
            competitor_comparison TEXT,
            thumbnail_scores TEXT
        )
    """)
    
    conn.commit()
    return conn


def save_analysis(
    video_name: str,
    niche: str,
    seo_score: int,
    viral_timestamp: str,
    sentiment_avg: float,
    hook_quality: str,
    keyword_density: float,
    transcript: str,
    visual_summary: str,
    recommendations: str,
    competitor_comparison: Optional[str] = None,
    thumbnail_scores: Optional[Dict] = None
) -> int:
    """
    Save a new video analysis to the database.
    
    Args:
        video_name: Name of the analyzed video file
        niche: Target keyword/niche
        seo_score: SEO relevance score (0-100)
        viral_timestamp: Timestamp of potential viral clip (e.g., "1:23-1:53")
        sentiment_avg: Average sentiment polarity
        hook_quality: Assessment of the video hook
        keyword_density: Percentage of keyword density
        transcript: Full audio transcript
        visual_summary: Visual analysis summary
        recommendations: Top 3 actionable tips
        competitor_comparison: Optional comparison analysis
        thumbnail_scores: Optional dict of thumbnail scores
        
    Returns:
        int: ID of the inserted record
    """
    conn = init_database()
    cursor = conn.cursor()
    
    # Convert thumbnail_scores dict to JSON string
    thumbnail_scores_json = json.dumps(thumbnail_scores) if thumbnail_scores else None
    
    cursor.execute("""
        INSERT INTO analyses (
            video_name, niche, seo_score, viral_timestamp, sentiment_avg,
            hook_quality, keyword_density, transcript, visual_summary,
            recommendations, competitor_comparison, thumbnail_scores
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        video_name, niche, seo_score, viral_timestamp, sentiment_avg,
        hook_quality, keyword_density, transcript, visual_summary,
        recommendations, competitor_comparison, thumbnail_scores_json
    ))
    
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    
    return record_id


def get_all_analyses(limit: int = 50) -> List[Dict]:
    """
    Retrieve all analyses from the database, ordered by date (newest first).
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        List of dictionaries containing analysis data
    """
    conn = init_database()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            id, video_name, niche, seo_score, viral_timestamp,
            sentiment_avg, hook_quality, keyword_density, analysis_date,
            transcript, visual_summary, recommendations,
            competitor_comparison, thumbnail_scores
        FROM analyses
        ORDER BY analysis_date DESC
        LIMIT ?
    """, (limit,))
    
    columns = [
        'id', 'video_name', 'niche', 'seo_score', 'viral_timestamp',
        'sentiment_avg', 'hook_quality', 'keyword_density', 'analysis_date',
        'transcript', 'visual_summary', 'recommendations',
        'competitor_comparison', 'thumbnail_scores'
    ]
    
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to list of dictionaries
    analyses = []
    for row in rows:
        analysis = dict(zip(columns, row))
        # Parse JSON thumbnail_scores back to dict
        if analysis['thumbnail_scores']:
            analysis['thumbnail_scores'] = json.loads(analysis['thumbnail_scores'])
        analyses.append(analysis)
    
    return analyses


def get_analysis_by_id(analysis_id: int) -> Optional[Dict]:
    """
    Retrieve a specific analysis by its ID.
    
    Args:
        analysis_id: ID of the analysis record
        
    Returns:
        Dictionary containing analysis data, or None if not found
    """
    conn = init_database()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            id, video_name, niche, seo_score, viral_timestamp,
            sentiment_avg, hook_quality, keyword_density, analysis_date,
            transcript, visual_summary, recommendations,
            competitor_comparison, thumbnail_scores
        FROM analyses
        WHERE id = ?
    """, (analysis_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    columns = [
        'id', 'video_name', 'niche', 'seo_score', 'viral_timestamp',
        'sentiment_avg', 'hook_quality', 'keyword_density', 'analysis_date',
        'transcript', 'visual_summary', 'recommendations',
        'competitor_comparison', 'thumbnail_scores'
    ]
    
    analysis = dict(zip(columns, row))
    
    # Parse JSON thumbnail_scores back to dict
    if analysis['thumbnail_scores']:
        analysis['thumbnail_scores'] = json.loads(analysis['thumbnail_scores'])
    
    return analysis


def get_score_trend(niche: Optional[str] = None, limit: int = 10) -> List[Tuple[str, int]]:
    """
    Get SEO score trend over time, optionally filtered by niche.
    
    Args:
        niche: Optional niche filter
        limit: Maximum number of records to return
        
    Returns:
        List of tuples (date, score)
    """
    conn = init_database()
    cursor = conn.cursor()
    
    if niche:
        cursor.execute("""
            SELECT analysis_date, seo_score
            FROM analyses
            WHERE niche = ?
            ORDER BY analysis_date ASC
            LIMIT ?
        """, (niche, limit))
    else:
        cursor.execute("""
            SELECT analysis_date, seo_score
            FROM analyses
            ORDER BY analysis_date ASC
            LIMIT ?
        """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return rows


def get_statistics() -> Dict:
    """
    Get overall statistics from the database.
    
    Returns:
        Dictionary with statistics (total_analyses, avg_score, etc.)
    """
    conn = init_database()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            AVG(seo_score) as avg_score,
            MAX(seo_score) as max_score,
            MIN(seo_score) as min_score
        FROM analyses
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    return {
        'total_analyses': row[0] or 0,
        'avg_score': round(row[1], 1) if row[1] else 0,
        'max_score': row[2] or 0,
        'min_score': row[3] or 0
    }


def delete_analysis(analysis_id: int) -> bool:
    """
    Delete an analysis record by ID.
    
    Args:
        analysis_id: ID of the record to delete
        
    Returns:
        True if deleted successfully, False otherwise
    """
    conn = init_database()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    
    deleted = cursor.rowcount > 0
    conn.close()
    
    return deleted


def clear_all_analyses() -> int:
    """
    Clear all analyses from the database (use with caution!).
    
    Returns:
        Number of records deleted
    """
    conn = init_database()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM analyses")
    conn.commit()
    
    count = cursor.rowcount
    conn.close()
    
    return count


# Initialize database on module import
init_database()
