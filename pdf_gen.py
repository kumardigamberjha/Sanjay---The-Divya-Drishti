"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      Empire-Grade PDF Report Generator                       ║
║                   Professional Client-Ready PDF Reports                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module generates beautifully formatted PDF reports with branding,
sentiment graphs, and actionable recommendations.
"""

from fpdf import FPDF
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path
import tempfile


class SanjayaReport(FPDF):
    """Custom PDF class with Wings of AI branding."""
    
    def header(self):
        """Add branded header to each page."""
        # Wings of AI branding
        self.set_font('Arial', 'B', 20)
        self.set_text_color(139, 92, 246)  # Purple
        self.cell(0, 10, 'Wings of AI', 0, 0, 'L')
        
        # Sanjaya logo
        self.set_font('Arial', 'I', 12)
        self.set_text_color(251, 191, 36)  # Gold
        self.cell(0, 10, 'Sanjaya - Divya Drishti', 0, 1, 'R')
        
        # Line separator
        self.set_draw_color(139, 92, 246)
        self.set_line_width(0.5)
        self.line(10, 25, 200, 25)
        self.ln(5)
    
    def footer(self):
        """Add footer to each page."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        # Timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        self.cell(0, 10, f'Generated: {timestamp}', 0, 0, 'R')
    
    def chapter_title(self, title: str, icon: str = ''):
        """Add a chapter title with optional icon."""
        self.set_font('Arial', 'B', 16)
        self.set_text_color(139, 92, 246)
        full_title = f"{icon} {title}" if icon else title
        self.cell(0, 10, full_title, 0, 1, 'L')
        self.ln(3)
    
    def body_text(self, text: str, bold: bool = False):
        """Add body text."""
        self.set_font('Arial', 'B' if bold else '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, text)
        self.ln(2)
    
    def add_metric_box(self, label: str, value: str, color: tuple = (139, 92, 246)):
        """Add a colored metric box."""
        # Box background
        self.set_fill_color(*color)
        self.set_draw_color(*color)
        self.rect(self.get_x(), self.get_y(), 90, 20, 'DF')
        
        # Label
        self.set_font('Arial', 'B', 10)
        self.set_text_color(255, 255, 255)
        self.cell(90, 8, label, 0, 1, 'C')
        
        # Value
        self.set_font('Arial', 'B', 14)
        self.cell(90, 8, value, 0, 1, 'C')
        self.ln(5)


def generate_report(
    video_name: str,
    niche: str,
    seo_score: int,
    viral_clip: Optional[Dict],
    sentiment_summary: Dict,
    hook_quality: str,
    keyword_density: float,
    recommendations: str,
    sentiment_graph_path: Optional[str] = None,
    competitor_comparison: Optional[str] = None,
    thumbnail_scores: Optional[Dict] = None
) -> bytes:
    """
    Generate a professional PDF report.
    
    Args:
        video_name: Name of the analyzed video
        niche: Target keyword/niche
        seo_score: SEO relevance score (0-100)
        viral_clip: Viral clip detection data
        sentiment_summary: Sentiment analysis summary
        hook_quality: Hook quality assessment
        keyword_density: Keyword density percentage
        recommendations: Top 3 actionable tips
        sentiment_graph_path: Path to sentiment graph image
        competitor_comparison: Optional competitor analysis
        thumbnail_scores: Optional thumbnail scoring results
        
    Returns:
        PDF file as bytes
    """
    pdf = SanjayaReport()
    pdf.add_page()
    
    # ========================================================================
    # COVER / EXECUTIVE SUMMARY
    # ========================================================================
    pdf.chapter_title('Executive Summary', '')
    
    # Video info
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f'Video: {video_name}', 0, 1)
    pdf.cell(0, 8, f'Target Niche: {niche}', 0, 1)
    pdf.cell(0, 8, f'Analysis Date: {datetime.now().strftime("%B %d, %Y")}', 0, 1)
    pdf.ln(5)
    
    # SEO Score - Large display
    score_color = (16, 185, 129) if seo_score >= 70 else (245, 158, 11) if seo_score >= 50 else (239, 68, 68)
    pdf.add_metric_box('SEO Relevance Score', f'{seo_score}/100', score_color)
    
    # Viral Potential
    if viral_clip:
        viral_text = f"Viral Clip: {format_time(viral_clip['start_time'])} - {format_time(viral_clip['end_time'])}"
        pdf.set_font('Arial', 'B', 11)
        pdf.set_text_color(236, 72, 153)  # Pink
        pdf.cell(0, 8, viral_text, 0, 1)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, viral_clip['description'])
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 8, 'No high-intensity viral moment detected', 0, 1)
    
    pdf.ln(10)
    
    # ========================================================================
    # SENTIMENT ANALYSIS
    # ========================================================================
    pdf.chapter_title('Emotional Intelligence Analysis', '')
    
    # Sentiment metrics
    pdf.body_text(f"Overall Tone: {sentiment_summary['overall_tone'].title()}", bold=True)
    pdf.body_text(f"Average Polarity: {sentiment_summary['avg_polarity']:.2f} (-1 to 1)")
    pdf.body_text(f"Average Intensity: {sentiment_summary['avg_intensity']:.2f} (0 to 1)")
    pdf.body_text(f"Emotional Range: {sentiment_summary['emotional_range']:.2f}")
    pdf.ln(5)
    
    # Embed sentiment graph if available
    if sentiment_graph_path and Path(sentiment_graph_path).exists():
        pdf.chapter_title('Sentiment Over Time', '')
        pdf.image(sentiment_graph_path, x=10, w=190)
        pdf.ln(5)
    
    # ========================================================================
    # DETAILED ANALYSIS
    # ========================================================================
    pdf.add_page()
    pdf.chapter_title('Detailed SEO Analysis', '')
    
    # Hook Quality
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(139, 92, 246)
    pdf.cell(0, 8, 'Hook Quality (First 5 Seconds):', 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, hook_quality)
    pdf.ln(3)
    
    # Keyword Density
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(139, 92, 246)
    pdf.cell(0, 8, 'Keyword Density:', 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    density_text = f"{keyword_density:.2f}% - "
    if keyword_density < 0.5:
        density_text += "Too low. Keyword is underutilized."
    elif keyword_density > 3:
        density_text += "Too high. Risk of keyword stuffing."
    else:
        density_text += "Optimal range for SEO."
    pdf.multi_cell(0, 6, density_text)
    pdf.ln(5)
    
    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    pdf.chapter_title('Actionable Recommendations', '')
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(0, 0, 0)
    
    # Parse recommendations (assuming they're formatted as numbered list)
    for line in recommendations.split('\n'):
        if line.strip():
            pdf.multi_cell(0, 6, line.strip())
            pdf.ln(2)
    
    pdf.ln(5)
    
    # ========================================================================
    # COMPETITOR COMPARISON (if provided)
    # ========================================================================
    if competitor_comparison:
        pdf.add_page()
        pdf.chapter_title('Competitor Benchmarking', '')
        pdf.set_font('Arial', '', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, competitor_comparison)
        pdf.ln(5)
    
    # ========================================================================
    # THUMBNAIL ANALYSIS (if provided)
    # ========================================================================
    if thumbnail_scores:
        pdf.add_page()
        pdf.chapter_title('Thumbnail A/B Testing Results', '')
        
        for i, (thumb_name, scores) in enumerate(thumbnail_scores.items(), 1):
            pdf.set_font('Arial', 'B', 12)
            pdf.set_text_color(139, 92, 246)
            pdf.cell(0, 8, f'Thumbnail {i}: {thumb_name}', 0, 1)
            
            pdf.set_font('Arial', '', 10)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 5, scores)
            pdf.ln(3)
    
    # ========================================================================
    # FOOTER MESSAGE
    # ========================================================================
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 5, 
        "This report was generated by Sanjaya - Divya Drishti, powered by Wings of AI. "
        "All analysis is performed using local AI models for complete privacy and control."
    )
    
    # Return PDF as bytes
    # fpdf2's output() returns bytearray, convert to bytes
    return bytes(pdf.output())



def format_time(seconds: int) -> str:
    """Format seconds as MM:SS."""
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"
