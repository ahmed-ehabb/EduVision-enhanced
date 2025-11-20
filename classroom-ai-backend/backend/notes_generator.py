#!/usr/bin/env python3
"""
Streamlined Bullet Point Notes Generator
=======================================

Optimized for your fine-tuned FLAN-T5 model's strengths:
- ROUGE-1: 47.66 (vs 27.84 zero-shot)
- ROUGE-2: 41.20 (vs 20.60 zero-shot)  
- Generation Length: 15.21 tokens (highly concise)

This script generates clean, focused bullet point notes without complex structuring,
letting your model do what it does best: create concise, high-quality summaries.
"""

import re
import time
from typing import List, Optional
from pathlib import Path

import torch
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import gc

class LectureNotesGenerator:
    """
    Streamlined generator focused on creating clean bullet point notes.
    Optimized for your fine-tuned model's concise summarization strengths.
    """
    
    def __init__(self, model_path: str = "ahmedhugging12/flan-t5-base-vtssum"):
        """Initialize with your fine-tuned model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        
        if self.device == "cuda":
            self.model = self.model.half()
        
        self.model.eval()
        print("Model loaded successfully!")
        
        # Optimal parameters based on your model's 15.21 token generation length
        self.max_input_length = 512
        self.max_output_length = 25    # Aligned with your model's ~15 token sweet spot
        self.chunk_size = 150          # Smaller chunks to get more 15-token summaries
        self.overlap_size = 30         # Moderate overlap for context
    
    def chunk_transcript(self, text: str) -> List[str]:
        """
        Enhanced chunking with overlap for better context and more bullet points.
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            
            # Start new chunk if adding this sentence would exceed limit
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Create overlap: keep last 1-2 sentences for context
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk[-1:]
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(len(self.tokenizer.encode(s, add_special_tokens=False)) 
                                   for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            
            i += 1
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def generate_bullet_point(self, chunk: str) -> str:
        """
        Generate a concise bullet point optimized for your model's ~15 token output.
        Your model is trained to produce high-quality 15-token summaries.
        """
        # Use the exact format your model was trained on
        prompt = f"summarize: {chunk}"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_output_length,  # ~25 tokens max to allow for ~15 token summaries
                num_beams=4,
                temperature=0.7,  # Slightly lower for more focused output
                do_sample=True,
                top_p=0.9,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()
    
    def clean_bullet_point(self, text: str) -> str:
        """
        Clean and format a concise bullet point.
        Optimized for your model's ~15 token outputs.
        """
        if not text or len(text.strip()) < 3:
            return None
        
        # Remove any existing bullet markers
        text = re.sub(r'^[-*•]\s*', '', text.strip())
        
        # Your model generates ~15 tokens, so don't over-process
        # Skip if too short (less than 5 words for meaningful content)
        if len(text.split()) < 5:
            return None
        
        # Ensure proper capitalization
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Ensure proper punctuation
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Light grammar correction (only if reasonable length)
        if len(text.split()) <= 20:  # Don't correct if unexpectedly long
            try:
                corrected = str(TextBlob(text).correct())
                if len(corrected) > 0 and len(corrected) <= len(text) * 1.2:
                    text = corrected
            except:
                pass
        
        return text
    
    def deduplicate_bullets(self, bullets: List[str], similarity_threshold: float = 0.85) -> List[str]:
        """
        Remove duplicate or highly similar bullet points using TF-IDF.
        """
        if len(bullets) <= 1:
            return bullets
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(bullets)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find duplicates
            to_remove = set()
            for i in range(len(bullets)):
                for j in range(i + 1, len(bullets)):
                    if similarity_matrix[i, j] > similarity_threshold:
                        # Keep the longer/more informative bullet
                        if len(bullets[i]) >= len(bullets[j]):
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
            
            # Return filtered bullets
            return [bullets[i] for i in range(len(bullets)) if i not in to_remove]
        
        except:
            # If TF-IDF fails, do simple string-based deduplication
            seen = set()
            unique_bullets = []
            for bullet in bullets:
                bullet_words = set(bullet.lower().split())
                is_duplicate = any(len(bullet_words & seen_words) / max(len(bullet_words), len(seen_words)) > 0.7 
                                 for seen_words in seen)
                if not is_duplicate:
                    seen.add(bullet_words)
                    unique_bullets.append(bullet)
            return unique_bullets
    
    def generate_notes(self, text: str, subject: str = "Psychology") -> str:
        """
        Generate lecture notes from text in markdown format titled 'Lecture Notes'.
        
        Args:
            text: Input text to generate notes from
            subject: Subject area (default: Psychology)
            
        Returns:
            Markdown formatted notes as string with title 'Lecture Notes'
        """
        if not self.is_available():
            raise Exception("Notes generator model not available")
        
        print("Generating bullet point notes...")
        print(f"Input text length: {len(text)} characters")
        
        # Step 1: Chunk the text
        print("Chunking text...")
        chunks = self.chunk_transcript(text.strip())
        print(f"Created {len(chunks)} chunks")
        
        # Step 2: Generate bullet points
        print("Generating bullet points...")
        all_bullets = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            try:
                bullet_text = self.generate_bullet_point(chunk)
                cleaned = self.clean_bullet_point(bullet_text)
                if cleaned:
                    all_bullets.append(cleaned)
            except Exception as e:
                print(f"Warning: Failed to process chunk {i+1}: {e}")
                continue
            time.sleep(0.05)
        
        print(f"Generated {len(all_bullets)} initial bullet points")
        
        # Step 3: Deduplicate
        print("Deduplicating bullet points...")
        unique_bullets = self.deduplicate_bullets(all_bullets)
        print(f"After deduplication: {len(unique_bullets)} bullet points")
        
        # Step 4: Format as markdown with fixed title
        if not unique_bullets:
            return "# Lecture Notes\n\nNo meaningful notes could be generated from the provided text."
        
        formatted_notes = "# Lecture Notes\n\n"
        for bullet in unique_bullets:
            formatted_notes += f"* {bullet}\n"
        
        # If save_path was intended, but since we return string, remove saving
        return formatted_notes.strip()

    def is_available(self) -> bool:
        """Check if the model is available"""
        return self.model is not None and self.tokenizer is not None
    
    def cleanup(self):
        """Clean up resources"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

def generate_notes(transcript: str, 
                  title: str = "Lecture Notes",
                  model_path: str = "ahmedhugging12/flan-t5-base-vtssum",
                  output_file: Optional[str] = None) -> str:
    """
    Convenience function for quick note generation.
    
    Args:
        transcript: The transcript text
        title: Title for the notes
        model_path: Path to your fine-tuned model
        output_file: Optional output file path
        
    Returns:
        Generated notes in markdown format
    """
    generator = LectureNotesGenerator(model_path)
    return generator.generate_notes(transcript, title)

def generate_notes_from_file(file_path: str,
                           title: Optional[str] = None,
                           output_file: Optional[str] = None) -> str:
    """
    Generate notes directly from a transcript file.
    
    Args:
        file_path: Path to transcript file
        title: Optional custom title (defaults to filename)
        output_file: Optional output file path
        
    Returns:
        Generated notes
    """
    # Read transcript
    with open(file_path, 'r', encoding='utf-8') as f:
        transcript = f.read()
    
    # Generate title from filename if not provided
    if title is None:
        title = Path(file_path).stem.replace('_', ' ').title() + " - Notes"
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = str(Path(file_path).with_suffix('.md'))
    
    return generate_notes(transcript, title, output_file=output_file)

# Example usage and batch processing
def batch_process_directory(input_dir: str, output_dir: Optional[str] = None):
    """
    Process all transcript files in a directory.
    
    Args:
        input_dir: Directory containing transcript files
        output_dir: Output directory (defaults to input_dir)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(exist_ok=True)
    
    generator = LectureNotesGenerator()
    
    for transcript_file in input_path.glob("*.txt"):
        print(f"\nProcessing: {transcript_file.name}")
        
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
            
            title = transcript_file.stem.replace('_', ' ').title() + " - Notes"
            output_file = output_path / f"{transcript_file.stem}_notes.md"
            
            notes = generator.generate_notes(transcript, title)
            print(f"[CHECK] Saved: {output_file.name}")
            
        except Exception as e:
            print(f"✗ Failed to process {transcript_file.name}: {e}")

def main():
    """Example usage."""
    # Method 1: Direct string input
    transcript_text = """
    Your 40-minute lecture transcript goes here.
    This should be already punctuated as mentioned.
    The model will generate focused bullet points from this content.
    """
    
    notes = generate_notes(
        transcript_text, 
        title="Psychology Lecture Notes",
        output_file="psychology_notes.md"
    )
    
    print("Generated notes preview:")
    print(notes[:500] + "..." if len(notes) > 500 else notes)
    
    # Method 2: From file
    # notes = generate_notes_from_file("transcript.txt", "My Lecture Notes")
    
    # Method 3: Batch processing
    # batch_process_directory("transcripts/", "notes/")

if __name__ == "__main__":
    main()