"""
Report Generator - Professional Lecture Analysis Reports
========================================================

Generates comprehensive PDF and HTML reports from teacher module outputs.

Features:
- PDF generation with ReportLab
- HTML generation with templates
- Charts and visualizations
- Professional formatting
- Metadata and timestamps
"""

import os
import re
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Charts
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from io import BytesIO

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate professional lecture analysis reports.

    Supports:
    - PDF export with charts
    - HTML export
    - Customizable templates
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize report generator.

        Args:
            output_dir: Directory for output files (default: ./reports)
        """
        self.output_dir = output_dir or os.path.join(os.getcwd(), "reports")
        os.makedirs(self.output_dir, exist_ok=True)

        # ReportLab styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        logger.info(f"[ReportGen] Initialized (output: {self.output_dir})")

    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""

        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))

        # Subsection heading
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=colors.HexColor('#555555'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))

        # Body text (custom style)
        self.styles.add(ParagraphStyle(
            name='ReportBodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))

        # Bullet point
        self.styles.add(ParagraphStyle(
            name='ReportBullet',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            leftIndent=20,
            spaceAfter=4,
            fontName='Helvetica'
        ))

    def _create_engagement_chart(self, engagement_score: float) -> BytesIO:
        """
        Create engagement score gauge chart.

        Args:
            engagement_score: Engagement percentage (0-100)

        Returns:
            BytesIO object containing PNG image
        """
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})

        # Create gauge
        theta = [0, engagement_score * 3.6 * (3.14159 / 180)]  # Convert to radians
        r = [0.8, 0.8]

        # Color based on score
        if engagement_score >= 70:
            color = '#27AE60'  # Green
        elif engagement_score >= 50:
            color = '#F39C12'  # Orange
        else:
            color = '#E74C3C'  # Red

        ax.plot(theta, r, linewidth=40, color=color)
        ax.set_ylim(0, 1)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)

        # Add percentage text
        ax.text(0, 0, f'{engagement_score:.1f}%',
                ha='center', va='center', fontsize=24, fontweight='bold')

        # Save to BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
        plt.close()
        buf.seek(0)

        return buf

    def _create_content_coverage_chart(self,
                                      covered_topics: int,
                                      total_topics: int) -> BytesIO:
        """
        Create content coverage pie chart.

        Args:
            covered_topics: Number of covered topics
            total_topics: Total number of topics

        Returns:
            BytesIO object containing PNG image
        """
        fig, ax = plt.subplots(figsize=(5, 5))

        uncovered = total_topics - covered_topics
        sizes = [covered_topics, uncovered]
        colors_list = ['#3498DB', '#ECF0F1']
        labels = [f'Covered ({covered_topics})', f'Not Covered ({uncovered})']

        ax.pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12})
        ax.axis('equal')

        # Save to BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
        plt.close()
        buf.seek(0)

        return buf

    def generate_pdf_report(self,
                           results: Dict[str, Any],
                           lecture_title: str = "Lecture Analysis",
                           output_filename: Optional[str] = None) -> str:
        """
        Generate PDF report from teacher module results.

        Args:
            results: Dictionary with all module outputs
            lecture_title: Title of the lecture
            output_filename: Optional custom filename

        Returns:
            Path to generated PDF file
        """
        logger.info(f"[ReportGen] Generating PDF report: {lecture_title}")

        # Generate filename
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"lecture_report_{timestamp}.pdf"

        output_path = os.path.join(self.output_dir, output_filename)

        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        # Build story (content)
        story = []

        # Title page
        story.append(Paragraph(lecture_title, self.styles['CustomTitle']))
        story.append(Spacer(1, 12))

        # Metadata
        meta_data = [
            ["Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Analysis Duration:", f"{results.get('total_time', 0):.1f} minutes"],
            ["Transcript Length:", f"{results.get('transcript_length', 0)} characters"]
        ]

        meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#555555')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#333333')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))

        story.append(meta_table)
        story.append(Spacer(1, 30))

        # Section 1: Engagement Analysis
        if 'engagement' in results:
            story.append(Paragraph("1. Engagement Analysis", self.styles['SectionHeading']))

            engagement = results['engagement']
            score = engagement.get('engagement_score', 0)

            # Add engagement chart
            try:
                chart_buf = self._create_engagement_chart(score)
                img = Image(chart_buf, width=4*inch, height=2*inch)
                story.append(img)
                story.append(Spacer(1, 12))
            except Exception as e:
                logger.warning(f"[ReportGen] Failed to create engagement chart: {e}")

            # Engagement metrics
            story.append(Paragraph(f"<b>Overall Engagement Score:</b> {score:.2f}%",
                                 self.styles['ReportBodyText']))
            story.append(Spacer(1, 6))

            # Detailed metrics
            if 'detailed_scores' in engagement:
                story.append(Paragraph("<b>Detailed Metrics:</b>", self.styles['SubsectionHeading']))

                metrics = engagement['detailed_scores']
                metrics_data = [
                    ["Metric", "Score"],
                    ["Active Learning", f"{metrics.get('active_learning', 0):.2f}%"],
                    ["Clarity", f"{metrics.get('clarity', 0):.2f}%"],
                    ["Interaction", f"{metrics.get('interaction', 0):.2f}%"],
                    ["Enthusiasm", f"{metrics.get('enthusiasm', 0):.2f}%"]
                ]

                metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))

                story.append(metrics_table)

            # Detailed segment engagement
            if 'segments' in engagement and engagement['segments']:
                story.append(Spacer(1, 12))
                story.append(Paragraph("<b>Segment-Level Engagement:</b>", self.styles['SubsectionHeading']))
                story.append(Spacer(1, 6))

                # Create table with segment engagement details
                eng_data = [['#', 'Timestamp', 'Text Preview', 'Score', 'Label']]
                for idx, seg in enumerate(engagement['segments'][:15], 1):  # Limit to first 15
                    timestamp = seg.get('timestamp', 'N/A')
                    text_preview = seg.get('text', '')[:50] + '...'
                    seg_score = seg.get('engagement_score', 0)
                    seg_label = seg.get('engagement_label', 'N/A')
                    eng_data.append([
                        str(idx),
                        timestamp,
                        text_preview,
                        f"{seg_score:.1f}%",
                        seg_label
                    ])

                eng_table = Table(eng_data, colWidths=[0.4*inch, 0.8*inch, 3*inch, 0.7*inch, 1.1*inch])
                eng_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                story.append(eng_table)

            story.append(Spacer(1, 20))

        # Section 2: Content Alignment
        if 'content_alignment' in results:
            story.append(Paragraph("2. Content Coverage", self.styles['SectionHeading']))

            alignment = results['content_alignment']
            coverage_pct = alignment.get('coverage_percentage', 0)
            covered = alignment.get('num_covered_topics', 0)
            total = alignment.get('num_total_topics', 0)

            # Add coverage chart
            if total > 0:
                try:
                    chart_buf = self._create_content_coverage_chart(covered, total)
                    img = Image(chart_buf, width=3.5*inch, height=3.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    logger.warning(f"[ReportGen] Failed to create coverage chart: {e}")

            story.append(Paragraph(
                f"<b>Coverage:</b> {covered}/{total} topics ({coverage_pct:.2f}%)",
                self.styles['ReportBodyText']
            ))
            story.append(Spacer(1, 6))

            # Overall alignment score (average similarity across all segments)
            if 'results' in alignment and alignment['results']:
                all_scores = [seg.get('similarity_score', 0) for seg in alignment['results']]
                overall_alignment_score = sum(all_scores) / len(all_scores) if all_scores else 0
                story.append(Paragraph(
                    f"<b>Overall Alignment Score:</b> {overall_alignment_score:.2f} (average similarity across all segments)",
                    self.styles['ReportBodyText']
                ))
                story.append(Spacer(1, 6))

                # Count segments by coverage type
                fully_covered = sum(1 for seg in alignment['results'] if seg.get('coverage_label') == 'Fully Covered')
                partially_covered = sum(1 for seg in alignment['results'] if seg.get('coverage_label') == 'Partially Covered')
                not_covered = sum(1 for seg in alignment['results'] if seg.get('coverage_label') == 'Not Covered')

                story.append(Paragraph(
                    f"<b>Segment Breakdown:</b> {fully_covered} Fully Covered, {partially_covered} Partially Covered, {not_covered} Not Covered",
                    self.styles['ReportBodyText']
                ))

            # Covered topics
            if 'covered_topics' in alignment and alignment['covered_topics']:
                story.append(Spacer(1, 8))
                story.append(Paragraph("<b>Covered Topics:</b>", self.styles['SubsectionHeading']))

                for topic in alignment['covered_topics'][:10]:  # Limit to 10
                    story.append(Paragraph(f"• {topic}", self.styles['ReportBullet']))

            # Detailed segment alignment information
            if 'results' in alignment and alignment['results']:
                story.append(Spacer(1, 12))
                story.append(Paragraph("<b>Detailed Segment Analysis:</b>", self.styles['SubsectionHeading']))
                story.append(Spacer(1, 6))

                # Create table with segment details (show more segments including partially covered)
                segment_data = [['#', 'Transcript Segment', 'Coverage', 'Score']]
                for idx, seg in enumerate(alignment['results'][:30], 1):  # Show first 30 segments
                    segment_text = seg.get('transcript_segment', '')[:60] + '...'
                    coverage_label = seg.get('coverage_label', 'N/A')
                    similarity_score = seg.get('similarity_score', 0)
                    segment_data.append([
                        str(idx),
                        segment_text,
                        coverage_label,
                        f"{similarity_score:.2f}"
                    ])

                seg_table = Table(segment_data, colWidths=[0.5*inch, 3.5*inch, 1.2*inch, 0.8*inch])
                seg_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                story.append(seg_table)

            story.append(Spacer(1, 20))

        # Section 3: Lecture Notes
        if 'notes' in results:
            story.append(Paragraph("3. Lecture Notes", self.styles['SectionHeading']))

            notes = results['notes']
            bullet_points = notes.get('bullet_points', [])

            story.append(Paragraph(
                f"<b>Total Notes:</b> {len(bullet_points)}",
                self.styles['ReportBodyText']
            ))
            story.append(Spacer(1, 12))

            for i, bullet in enumerate(bullet_points, 1):
                story.append(Paragraph(f"{i}. {bullet}", self.styles['ReportBullet']))

            story.append(Spacer(1, 20))

        # Section 4: Quiz Questions
        if 'quiz' in results:
            story.append(PageBreak())  # New page for quiz
            story.append(Paragraph("4. Assessment Questions", self.styles['SectionHeading']))

            quiz = results['quiz']

            # Check if quiz was skipped due to memory
            if quiz.get('skipped'):
                story.append(Paragraph(
                    f"<b>Status:</b> Quiz generation was skipped",
                    self.styles['ReportBodyText']
                ))
                story.append(Spacer(1, 6))
                reason = quiz.get('reason', 'Unknown reason')
                story.append(Paragraph(
                    f"<b>Reason:</b> {reason}",
                    self.styles['ReportBodyText']
                ))
                story.append(Spacer(1, 20))
            else:
                # Multiple Choice Questions
                mcq_questions = quiz.get('mcq_questions', quiz.get('questions', []))
                num_mcq = quiz.get('num_mcq', len(mcq_questions))

                if mcq_questions:
                    story.append(Paragraph("4.1 Multiple Choice Questions", self.styles['SubsectionHeading']))
                    story.append(Paragraph(
                        f"<b>Total MCQ:</b> {num_mcq}",
                        self.styles['ReportBodyText']
                    ))
                    story.append(Spacer(1, 12))

                    for i, q in enumerate(mcq_questions, 1):
                        # Question
                        story.append(Paragraph(
                            f"<b>Question {i}:</b> {q.get('question', '')}",
                            self.styles['SubsectionHeading']
                        ))

                        # Options
                        options = q.get('options', {})
                        if isinstance(options, dict):
                            # Dict format: {'a': '...', 'b': '...'}
                            for opt_letter in ['a', 'b', 'c', 'd']:
                                if opt_letter in options:
                                    story.append(Paragraph(
                                        f"{opt_letter.upper()}. {options[opt_letter]}",
                                        self.styles['ReportBullet']
                                    ))
                        else:
                            # List format
                            for opt_idx, option in enumerate(options):
                                marker = chr(65 + opt_idx)  # A, B, C, D
                                story.append(Paragraph(f"{marker}. {option}", self.styles['ReportBullet']))

                        # Correct answer
                        correct = q.get('correct_answer', '')
                        story.append(Paragraph(
                            f"<b>Correct Answer:</b> {correct.upper()}",
                            self.styles['ReportBodyText']
                        ))

                        # Explanation
                        explanation = q.get('explanation', '')
                        if explanation:
                            story.append(Paragraph(
                                f"<b>Explanation:</b> {explanation}",
                                self.styles['ReportBodyText']
                            ))

                        story.append(Spacer(1, 16))

                # Open-Ended Questions (NEW)
                open_ended_text = quiz.get('open_ended_questions', '')
                num_open_ended = quiz.get('num_open_ended', 0)

                if open_ended_text and num_open_ended > 0:
                    story.append(Spacer(1, 20))
                    story.append(Paragraph("4.2 Open-Ended Questions", self.styles['SubsectionHeading']))
                    story.append(Paragraph(
                        f"<b>Total Open-Ended:</b> {num_open_ended}",
                        self.styles['ReportBodyText']
                    ))
                    story.append(Spacer(1, 12))

                    # Parse and format open-ended questions
                    # DEBUG: Log the raw text to help diagnose parsing issues
                    logger.info(f"[ReportGen] Parsing {num_open_ended} open-ended questions from text (length: {len(open_ended_text)} chars)")
                    logger.info(f"[ReportGen] First 200 chars: {open_ended_text[:200]}")

                    # Split by "**Question N**" or "Question N:" patterns
                    question_blocks = re.split(r'(?=(?:\*\*Question \d+\*\*|Question \d+:|^\d+\.))', open_ended_text, flags=re.MULTILINE)
                    logger.info(f"[ReportGen] Split into {len(question_blocks)} blocks")

                    for block in question_blocks:
                        block = block.strip()
                        if not block:
                            continue

                        # Try to match "**Question N**" format (markdown bold)
                        q_match = re.match(r'\*\*Question (\d+)\*\*\s*(.+?)(?=\*\*Sample Answer\*\*|\n\n\*\*Question|\n\n|$)', block, re.DOTALL)

                        # If that fails, try "Question N:" format
                        if not q_match:
                            q_match = re.match(r'Question (\d+):\s*(.+?)(?=\nSample Answer:|\n\n|$)', block, re.DOTALL)

                        # If that fails, try numbered format "N."
                        if not q_match:
                            q_match = re.match(r'(\d+)\.\s*(.+?)(?=\nSample Answer:|\n\n|$)', block, re.DOTALL)

                        if q_match:
                            q_num = q_match.group(1)
                            q_text = q_match.group(2).strip()

                            # Add question
                            story.append(Paragraph(
                                f"<b>Question {q_num}:</b> {q_text}",
                                self.styles['SubsectionHeading']
                            ))
                            story.append(Spacer(1, 6))

                            # Extract sample answer if present (handle both markdown and plain formats)
                            ans_match = re.search(r'(?:\*\*Sample Answer\*\*|Sample Answer):\s*(.+?)(?=(?:\*\*Question \d+\*\*|Question \d+:|\d+\.)|$)', block, re.DOTALL)
                            if ans_match:
                                sample_answer = ans_match.group(1).strip()
                                story.append(Paragraph(
                                    f"<b>Sample Answer:</b>",
                                    self.styles['ReportBodyText']
                                ))
                                story.append(Spacer(1, 4))
                                story.append(Paragraph(
                                    sample_answer,
                                    self.styles['ReportBullet']
                                ))

                            story.append(Spacer(1, 16))

                story.append(Spacer(1, 20))

        # Section 5: Translation (if available)
        if 'translation' in results:
            story.append(PageBreak())  # New page for translation
            story.append(Paragraph("5. Arabic Translation", self.styles['SectionHeading']))

            translation = results['translation']
            arabic_text = translation.get('arabic_text', '')

            if arabic_text:
                # Note: ReportLab requires Arabic font support
                # For now, just indicate translation is available
                story.append(Paragraph(
                    f"<b>Translation Length:</b> {len(arabic_text)} characters",
                    self.styles['ReportBodyText']
                ))
                story.append(Paragraph(
                    "Arabic translation has been generated. See separate file for full text.",
                    self.styles['ReportBodyText']
                ))

        # Footer
        story.append(PageBreak())
        story.append(Paragraph("Generated by EduVision Teacher Module V2",
                             self.styles['ReportBodyText']))
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                             self.styles['ReportBodyText']))

        # Build PDF
        doc.build(story)

        logger.info(f"[ReportGen] PDF report saved: {output_path}")
        return output_path

    def generate_html_report(self,
                            results: Dict[str, Any],
                            lecture_title: str = "Lecture Analysis",
                            output_filename: Optional[str] = None) -> str:
        """
        Generate HTML report from teacher module results.

        Args:
            results: Dictionary with all module outputs
            lecture_title: Title of the lecture
            output_filename: Optional custom filename

        Returns:
            Path to generated HTML file
        """
        logger.info(f"[ReportGen] Generating HTML report: {lecture_title}")

        # Generate filename
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"lecture_report_{timestamp}.html"

        output_path = os.path.join(self.output_dir, output_filename)

        # Build HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{lecture_title} - Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2C3E50;
            border-bottom: 3px solid #3498DB;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495E;
            margin-top: 30px;
            border-bottom: 2px solid #ECF0F1;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #555;
        }}
        .meta {{
            background: #ECF0F1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .meta-item {{
            margin: 5px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 15px 25px;
            background: #3498DB;
            color: white;
            border-radius: 5px;
            font-size: 18px;
            font-weight: bold;
        }}
        .metric.high {{
            background: #27AE60;
        }}
        .metric.medium {{
            background: #F39C12;
        }}
        .metric.low {{
            background: #E74C3C;
        }}
        .bullet-points {{
            list-style-type: none;
            padding-left: 0;
        }}
        .bullet-points li {{
            padding: 8px;
            margin: 5px 0;
            background: #F8F9FA;
            border-left: 4px solid #3498DB;
            padding-left: 15px;
        }}
        .question {{
            background: #F8F9FA;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 4px solid #3498DB;
        }}
        .question.open-ended {{
            border-left-color: #9B59B6;
        }}
        .question-text {{
            font-size: 15px;
            line-height: 1.6;
            margin: 10px 0;
            color: #2C3E50;
        }}
        .sample-answer {{
            margin-top: 15px;
            padding: 15px;
            background: #E8F5E9;
            border-radius: 5px;
            border-left: 3px solid #27AE60;
        }}
        .sample-answer strong {{
            color: #27AE60;
            display: block;
            margin-bottom: 8px;
        }}
        .sample-answer p {{
            margin: 0;
            line-height: 1.7;
            color: #333;
        }}
        .options {{
            list-style-type: none;
            padding-left: 0;
        }}
        .options li {{
            padding: 8px;
            margin: 5px 0;
            background: white;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}
        .correct-answer {{
            color: #27AE60;
            font-weight: bold;
            margin-top: 10px;
        }}
        .explanation {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{lecture_title}</h1>

        <div class="meta">
            <div class="meta-item"><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <div class="meta-item"><strong>Analysis Duration:</strong> {results.get('total_time', 0):.1f} minutes</div>
            <div class="meta-item"><strong>Transcript Length:</strong> {results.get('transcript_length', 0)} characters</div>
        </div>
"""

        # Engagement section
        if 'engagement' in results:
            engagement = results['engagement']
            score = engagement.get('engagement_score', 0)

            score_class = 'high' if score >= 70 else ('medium' if score >= 50 else 'low')

            html += f"""
        <h2>1. Engagement Analysis</h2>
        <div class="metric {score_class}">{score:.2f}% Engagement</div>
"""

            if 'detailed_scores' in engagement:
                metrics = engagement['detailed_scores']
                html += """
        <h3>Detailed Metrics</h3>
        <ul class="bullet-points">
"""
                for metric_name, metric_value in metrics.items():
                    html += f"            <li><strong>{metric_name.replace('_', ' ').title()}:</strong> {metric_value:.2f}%</li>\n"

                html += "        </ul>\n"

            # Engagement segment details
            if 'segments' in engagement and engagement['segments']:
                html += """
        <h3>Segment-Level Engagement</h3>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <tr style="background-color: #777; color: white;">
                <th style="padding: 8px; border: 1px solid #ddd;">#</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Timestamp</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Text Preview</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Score</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Label</th>
            </tr>
"""
                for idx, seg in enumerate(engagement['segments'][:15], 1):
                    timestamp = seg.get('timestamp', 'N/A')
                    text_preview = seg.get('text', '')[:60] + '...'
                    seg_score = seg.get('engagement_score', 0)
                    seg_label = seg.get('engagement_label', 'N/A')
                    html += f"""
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">{idx}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{timestamp}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{text_preview}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{seg_score:.1f}%</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{seg_label}</td>
            </tr>
"""
                html += "        </table>\n"

        # Content alignment section
        if 'content_alignment' in results:
            alignment = results['content_alignment']
            coverage_pct = alignment.get('coverage_percentage', 0)
            covered = alignment.get('num_covered_topics', 0)
            total = alignment.get('num_total_topics', 0)

            html += f"""
        <h2>2. Content Coverage</h2>
        <div class="metric">{covered}/{total} Topics ({coverage_pct:.2f}%)</div>
"""

            # Overall alignment score and segment breakdown
            if 'results' in alignment and alignment['results']:
                all_scores = [seg.get('similarity_score', 0) for seg in alignment['results']]
                overall_alignment_score = sum(all_scores) / len(all_scores) if all_scores else 0

                fully_covered = sum(1 for seg in alignment['results'] if seg.get('coverage_label') == 'Fully Covered')
                partially_covered = sum(1 for seg in alignment['results'] if seg.get('coverage_label') == 'Partially Covered')
                not_covered = sum(1 for seg in alignment['results'] if seg.get('coverage_label') == 'Not Covered')

                html += f"""
        <p><strong>Overall Alignment Score:</strong> {overall_alignment_score:.2f} (average similarity across all segments)</p>
        <p><strong>Segment Breakdown:</strong> {fully_covered} Fully Covered, {partially_covered} Partially Covered, {not_covered} Not Covered</p>
"""

            if 'covered_topics' in alignment and alignment['covered_topics']:
                html += """
        <h3>Covered Topics</h3>
        <ul class="bullet-points">
"""
                for topic in alignment['covered_topics']:
                    html += f"            <li>{topic}</li>\n"

                html += "        </ul>\n"

            # Content alignment segment details
            if 'results' in alignment and alignment['results']:
                html += """
        <h3>Detailed Segment Analysis</h3>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <tr style="background-color: #777; color: white;">
                <th style="padding: 8px; border: 1px solid #ddd;">#</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Transcript Segment</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Coverage</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Score</th>
            </tr>
"""
                for idx, seg in enumerate(alignment['results'][:30], 1):  # Show first 30
                    segment_text = seg.get('transcript_segment', '')[:80] + '...'
                    coverage_label = seg.get('coverage_label', 'N/A')
                    similarity_score = seg.get('similarity_score', 0)
                    html += f"""
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px; border: 1px solid #ddd;">{idx}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{segment_text}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{coverage_label}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{similarity_score:.2f}</td>
            </tr>
"""
                html += "        </table>\n"

        # Notes section
        if 'notes' in results:
            notes = results['notes']
            bullet_points = notes.get('bullet_points', [])

            html += f"""
        <h2>3. Lecture Notes</h2>
        <p><strong>Total Notes:</strong> {len(bullet_points)}</p>
        <ul class="bullet-points">
"""
            for bullet in bullet_points:
                html += f"            <li>{bullet}</li>\n"

            html += "        </ul>\n"

        # Quiz section
        if 'quiz' in results:
            quiz = results['quiz']

            html += """
        <h2>4. Assessment Questions</h2>
"""

            # Check if quiz was skipped
            if quiz.get('skipped'):
                reason = quiz.get('reason', 'Unknown reason')
                html += f"""
        <p><strong>Status:</strong> Quiz generation was skipped</p>
        <p><strong>Reason:</strong> {reason}</p>
"""
            else:
                # Multiple Choice Questions
                mcq_questions = quiz.get('mcq_questions', quiz.get('questions', []))
                num_mcq = quiz.get('num_mcq', len(mcq_questions))

                if mcq_questions:
                    html += f"""
        <h3>4.1 Multiple Choice Questions</h3>
        <p><strong>Total MCQ:</strong> {num_mcq}</p>
"""

                    for i, q in enumerate(mcq_questions, 1):
                        html += f"""
        <div class="question">
            <h4>Question {i}: {q.get('question', '')}</h4>
            <ul class="options">
"""
                        options = q.get('options', {})
                        if isinstance(options, dict):
                            # Dict format: {'a': '...', 'b': '...'}
                            for opt_letter in ['a', 'b', 'c', 'd']:
                                if opt_letter in options:
                                    html += f"                <li><strong>{opt_letter.upper()}.</strong> {options[opt_letter]}</li>\n"
                        else:
                            # List format
                            for opt_idx, option in enumerate(options):
                                marker = chr(65 + opt_idx)  # A, B, C, D
                                html += f"                <li><strong>{marker}.</strong> {option}</li>\n"

                        html += f"""            </ul>
            <div class="correct-answer">Correct Answer: {q.get('correct_answer', '').upper()}</div>
"""

                        explanation = q.get('explanation', '')
                        if explanation:
                            html += f"            <div class=\"explanation\">Explanation: {explanation}</div>\n"

                        html += "        </div>\n"

                # Open-Ended Questions (NEW)
                open_ended_text = quiz.get('open_ended_questions', '')
                num_open_ended = quiz.get('num_open_ended', 0)

                if open_ended_text and num_open_ended > 0:
                    html += f"""
        <h3>4.2 Open-Ended Questions</h3>
        <p><strong>Total Open-Ended:</strong> {num_open_ended}</p>
"""

                    # Parse and format open-ended questions (handle markdown format)
                    question_blocks = re.split(r'(?=(?:\*\*Question \d+\*\*|Question \d+:|^\d+\.))', open_ended_text, flags=re.MULTILINE)

                    for block in question_blocks:
                        block = block.strip()
                        if not block:
                            continue

                        # Try to match "**Question N**" format (markdown bold)
                        q_match = re.match(r'\*\*Question (\d+)\*\*\s*(.+?)(?=\*\*Sample Answer\*\*|\n\n\*\*Question|\n\n|$)', block, re.DOTALL)

                        # If that fails, try "Question N:" format
                        if not q_match:
                            q_match = re.match(r'Question (\d+):\s*(.+?)(?=\nSample Answer:|\n\n|$)', block, re.DOTALL)

                        # If that fails, try numbered format "N."
                        if not q_match:
                            q_match = re.match(r'(\d+)\.\s*(.+?)(?=\nSample Answer:|\n\n|$)', block, re.DOTALL)

                        if q_match:
                            q_num = q_match.group(1)
                            q_text = q_match.group(2).strip()

                            html += f"""
        <div class="question open-ended">
            <h4>Question {q_num}:</h4>
            <p class="question-text">{q_text}</p>
"""

                            # Extract sample answer if present (handle both markdown and plain formats)
                            ans_match = re.search(r'(?:\*\*Sample Answer\*\*|Sample Answer):\s*(.+?)(?=(?:\*\*Question \d+\*\*|Question \d+:|\d+\.)|$)', block, re.DOTALL)
                            if ans_match:
                                sample_answer = ans_match.group(1).strip()
                                html += f"""
            <div class="sample-answer">
                <strong>Sample Answer:</strong>
                <p>{sample_answer}</p>
            </div>
"""

                            html += "        </div>\n"

        # Translation section
        if 'translation' in results:
            translation = results['translation']
            arabic_text = translation.get('arabic_text', '')

            html += f"""
        <h2>5. Arabic Translation</h2>
        <p><strong>Translation Length:</strong> {len(arabic_text)} characters</p>
        <p>Arabic translation has been generated. See separate file for full text.</p>
"""

        # Footer
        html += f"""
        <div class="footer">
            <p>Generated by EduVision Teacher Module V2</p>
            <p>Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"[ReportGen] HTML report saved: {output_path}")
        return output_path

    def generate_reports(self,
                        results: Dict[str, Any],
                        lecture_title: str = "Lecture Analysis",
                        formats: List[str] = ['pdf', 'html']) -> Dict[str, str]:
        """
        Generate multiple report formats.

        Args:
            results: Dictionary with all module outputs
            lecture_title: Title of the lecture
            formats: List of formats to generate ('pdf', 'html')

        Returns:
            Dictionary mapping format to output path
        """
        output_paths = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if 'pdf' in formats:
            try:
                pdf_path = self.generate_pdf_report(
                    results,
                    lecture_title,
                    f"lecture_report_{timestamp}.pdf"
                )
                output_paths['pdf'] = pdf_path
            except Exception as e:
                logger.error(f"[ReportGen] Failed to generate PDF: {e}")

        if 'html' in formats:
            try:
                html_path = self.generate_html_report(
                    results,
                    lecture_title,
                    f"lecture_report_{timestamp}.html"
                )
                output_paths['html'] = html_path
            except Exception as e:
                logger.error(f"[ReportGen] Failed to generate HTML: {e}")

        return output_paths


# Convenience function
def generate_report(results: Dict[str, Any],
                   lecture_title: str = "Lecture Analysis",
                   output_dir: Optional[str] = None,
                   formats: List[str] = ['pdf', 'html']) -> Dict[str, str]:
    """
    Generate lecture analysis reports.

    Args:
        results: Dictionary with all module outputs
        lecture_title: Title of the lecture
        output_dir: Output directory (default: ./reports)
        formats: List of formats ('pdf', 'html')

    Returns:
        Dictionary mapping format to output path
    """
    generator = ReportGenerator(output_dir)
    return generator.generate_reports(results, lecture_title, formats)


# Test function
def test_report_generator():
    """Test report generator with sample data."""

    # Sample results
    sample_results = {
        'total_time': 32.5,
        'transcript_length': 6356,
        'engagement': {
            'engagement_score': 65.38,
            'detailed_scores': {
                'active_learning': 72.5,
                'clarity': 68.2,
                'interaction': 55.3,
                'enthusiasm': 65.9
            }
        },
        'content_alignment': {
            'coverage_percentage': 11.54,
            'num_covered_topics': 3,
            'num_total_topics': 26,
            'covered_topics': [
                'Introduction to Psychology',
                'Learning Theories',
                'Behavioral Psychology'
            ]
        },
        'notes': {
            'bullet_points': [
                'Psychology is the scientific study of mind and behavior.',
                'Behavioral psychology focuses on observable behaviors.',
                'Learning can occur through reinforcement and punishment.',
                'Cognitive psychology examines mental processes.'
            ]
        },
        'quiz': {
            'questions': [
                {
                    'question': 'What is the primary focus of behavioral psychology?',
                    'options': [
                        'Mental processes',
                        'Observable behaviors',
                        'Emotional states',
                        'Unconscious mind'
                    ],
                    'correct_answer': 'B',
                    'explanation': 'Behavioral psychology focuses on observable behaviors rather than internal mental states.'
                },
                {
                    'question': 'Who is considered the father of behaviorism?',
                    'options': [
                        'Sigmund Freud',
                        'B.F. Skinner',
                        'John Watson',
                        'Carl Rogers'
                    ],
                    'correct_answer': 'C',
                    'explanation': 'John Watson is considered the father of behaviorism, though B.F. Skinner made major contributions.'
                }
            ]
        },
        'translation': {
            'arabic_text': 'النص المترجم...'
        }
    }

    # Generate reports
    print("[INFO] Testing report generator...")

    generator = ReportGenerator()
    paths = generator.generate_reports(
        sample_results,
        "Introduction to Psychology - Lecture 1",
        formats=['pdf', 'html']
    )

    print(f"\n[OK] Reports generated:")
    for format_type, path in paths.items():
        print(f"  {format_type.upper()}: {path}")


if __name__ == "__main__":
    test_report_generator()
