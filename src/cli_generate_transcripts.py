#!/usr/bin/env python3
# =========================================================================
# MLX-YouTubeScribe - CLI Version
# Objective: Generate transcripts from YouTube videos using local Whisper models with MLX acceleration
# =========================================================================
# Author: spidernic
# Created: May 2025
# =========================================================================
# Copyright 2025 spidernic
# [Apache License 2.0 details]

import os
import json
import yt_dlp
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from typing import Any, List, Tuple, Dict, Optional
from scipy.io import wavfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from dataclasses import dataclass

import torch
modelo1 = "openai/whisper-large-v3-turbo"
modelo2 = "openai/whisper-tiny.en"
@dataclass
class AudioTranscriber:
    """Audio transcription using Whisper"""
    processor: WhisperProcessor
    model: WhisperForConditionalGeneration
    
    def __init__(self):
        # Initialize with English-only model
        self.processor = WhisperProcessor.from_pretrained(modelo1)
        self.model = WhisperForConditionalGeneration.from_pretrained(modelo1)
        
    def transcribe(self, audio_features: mx.array) -> str:
        """Transcribe audio using Whisper"""
        try:
            # Convert MLX array to numpy for processing
            audio_np = audio_features.tolist()
            if isinstance(audio_np, list):
                audio_np = np.array(audio_np)
            
            # Process audio features
            inputs = self.processor(
                audio_np,
                return_tensors="pt",
                sampling_rate=16000,
                return_attention_mask=True,
                language="en"  # Explicitly set English as target language
            )
            
            # Generate transcription using PyTorch
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_features,
                    attention_mask=inputs.attention_mask,
                    return_timestamps=True,
                    max_length=448,  # Limit output length
                    language='en'  # Force English translation
                )
            
            # Decode the output
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            return transcription
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return ""



def get_video_info(youtube_url: str, output_dir: str = None) -> Tuple[Optional[dict], Optional[str]]:
    """Get video information and download audio using yt-dlp"""
    try:
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',  # Choose best audio quality
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'quiet': False,  # Set to False for debugging
            'no_warnings': False  # Set to False for debugging
        }
        
        # If output_dir is provided, save WAV there, otherwise use temp file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Use sanitized title for output filename
            ydl_opts['outtmpl'] = os.path.join(output_dir, '%(title)s.%(ext)s')
            
            # First get info without downloading
            with yt_dlp.YoutubeDL({**ydl_opts, 'extract_flat': True}) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                if not info:
                    print("Failed to extract video info.")
                    return None, None
                
                # Sanitize title for filename
                title = info.get('title', '').replace('/', '_').replace(':', '_').replace('?', '_').replace('|', '_')
                expected_wav = os.path.join(output_dir, f"{title}.wav")
                
                # Now download with the same options
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
                
                if os.path.exists(expected_wav):
                    return info, expected_wav
                
                # Fallback: look for any WAV file if the expected one isn't found
                print(f"Expected WAV file {expected_wav} not found. Searching for any WAV file in {output_dir}...")
                for file in os.listdir(output_dir):
                    if file.endswith('.wav'):
                        wav_file = os.path.join(output_dir, file)
                        print(f"Found WAV file: {wav_file}")
                        return info, wav_file
                
                print(f"No WAV file found in {output_dir}.")
                return None, None
        else:
            # Use temporary file
            ydl_opts['outtmpl'] = 'temp_audio'
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                return info, 'temp_audio.wav'
    
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None, None

def process_audio_features(audio_data: np.ndarray, sample_rate: int) -> list[mx.array]:
    """Process audio data into features using MLX"""
    # Handle different audio channel configurations
    if len(audio_data.shape) == 1:
        # Already mono
        mono_audio = audio_data
    elif len(audio_data.shape) == 2:
        # Convert stereo to mono by averaging channels
        if audio_data.shape[1] == 2:  # Channels in second dimension
            mono_audio = np.mean(audio_data, axis=1)
        else:  # Channels in first dimension
            mono_audio = np.mean(audio_data, axis=0)
    else:
        raise ValueError(f"Unsupported audio shape: {audio_data.shape}")
    
    # Convert to float32 and normalize
    if mono_audio.dtype == np.int16:
        mono_audio = mono_audio.astype(np.float32) / 32768.0
    elif mono_audio.dtype == np.int32:
        mono_audio = mono_audio.astype(np.float32) / 2147483648.0
    elif mono_audio.dtype == np.float64:
        # Scale to [-1, 1] range
        max_val = np.max(np.abs(mono_audio))
        if max_val > 0:
            mono_audio = mono_audio.astype(np.float32) / max_val
        else:
            mono_audio = mono_audio.astype(np.float32)
    
    # Ensure we're working with float32
    mono_audio = mono_audio.astype(np.float32)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        # Calculate ratio for resampling
        ratio = 16000 / sample_rate
        new_length = int(len(mono_audio) * ratio)
        indices = np.linspace(0, len(mono_audio) - 1, new_length)
        mono_audio = np.interp(indices, np.arange(len(mono_audio)), mono_audio)
    
    # Split audio into 30-second chunks (16000 samples/sec * 30 sec = 480000 samples)
    chunk_size = 480000
    audio_chunks = []
    
    for i in range(0, len(mono_audio), chunk_size):
        chunk = mono_audio[i:i + chunk_size]
        # Pad last chunk if needed
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        audio_chunks.append(mx.array(chunk))
    
    return audio_chunks

def process_audio(audio_path: str) -> Tuple[str, dict]:
    """Process audio file using MLX and return a summary of its characteristics and detailed metrics"""
    try:
        # Load audio file
        sample_rate, audio_data = wavfile.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.float64:
            # Scale to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data.astype(np.float32) / max_val
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Calculate basic audio characteristics
        duration = len(audio_data) / sample_rate
        num_samples = len(audio_data)
        peak_amplitude = float(np.max(np.abs(audio_data)))
        rms = float(np.sqrt(np.mean(np.square(audio_data))))
        
        # Create detailed metrics dictionary
        metrics = {
            "duration": duration,
            "num_samples": num_samples,
            "sample_rate": sample_rate,
            "peak_amplitude": peak_amplitude,
            "rms": rms
        }
        
        # Create detailed audio summary
        audio_info = f"""Audio Analysis:
        - Duration: {duration:.2f} seconds
        - Number of samples: {num_samples}
        - Sample rate: {sample_rate} Hz
        - Peak Amplitude: {peak_amplitude:.4f}
        - RMS Energy: {rms:.4f}
        - Channel Processing: Converted to mono for analysis
        - Normalization: Applied based on bit depth"""
        
        return audio_info, metrics
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return "", {}
        return ""

def transcribe_audio(audio_features: list[mx.array]) -> str:
    """Transcribe audio using Whisper MLX"""
    transcriber = AudioTranscriber()
    transcriptions = []
    total_chunks = len(audio_features)
    
    print(f"\nProcessing {total_chunks} audio chunks...")
    print("-" * 50)
    
    for i, chunk in enumerate(audio_features, 1):
        print(f"\rTranscribing chunk {i}/{total_chunks}... ", end="")
        trans = transcriber.transcribe(chunk)
        if trans:
            # Clean up the transcription
            trans = trans.strip()
            # Remove leading/trailing quotes and periods
            trans = trans.strip('".')
            # Remove any duplicate spaces
            trans = ' '.join(trans.split())
            if trans:
                transcriptions.append(trans)
        
        # Show progress percentage
        progress = (i / total_chunks) * 100
        print(f"[{progress:3.0f}%]", end="")
    
    print("\n" + "-" * 50)
    
    # Join transcriptions with proper spacing and punctuation
    full_transcript = ". ".join(t for t in transcriptions if t)
    if full_transcript:
        full_transcript += "."
    
    return full_transcript



def analyze_youtube_video(youtube_url: str, output_dir: str = 'output') -> tuple[str, list]:
    """Analyze a YouTube video using yt-dlp and Whisper for transcription"""
    audio_path = None
    try:
        # Initialize transcriber
        transcriber = AudioTranscriber()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create audio subdirectory
        audio_dir = os.path.join(output_dir, 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        # Get video information and download audio
        info, audio_path = get_video_info(youtube_url, output_dir=audio_dir)
        if not info or not audio_path:
            raise ValueError("Could not fetch video information and audio")
        
        # Extract relevant information
        title = info.get('title', '').replace('/', '_')  # Sanitize filename
        description = info.get('description', '')
        duration = info.get('duration', 0)
        view_count = info.get('view_count', 0)
        
        # Process audio for analysis
        audio_analysis, audio_metrics = process_audio(audio_path)
        
        # Read audio for transcription
        sample_rate, audio_data = wavfile.read(audio_path)
        
        # Process audio features
        audio_features = process_audio_features(audio_data, sample_rate)
        
        # Generate transcript using Whisper
        print("\nGenerating transcript...")
        transcript = transcribe_audio(audio_features)
        
        if not transcript:
            raise ValueError("Failed to generate transcript")
        
        # Prepare output data
        output_data = {
            'video_info': {
                'title': title,
                'description': description,
                'duration': duration,
                'view_count': view_count,
                'url': youtube_url
            },
            'audio_analysis': audio_analysis,
            'transcript': transcript,
            'audio_file': os.path.basename(audio_path)
        }
        
        # Save to JSON file
        output_path = os.path.join(output_dir, f"{title}.json")
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Save to text file for better readability
        txt_path = os.path.join(output_dir, f"{title}.txt")
        with open(txt_path, 'w') as f:
            f.write(f"Video Analysis: {title}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Video Information:\n")
            f.write("-" * 20 + "\n")
            f.write(f"URL: {youtube_url}\n")
            f.write(f"Title: {title}\n")
            f.write(f"Duration: {duration} seconds\n")
            f.write(f"Views: {view_count}\n")
            f.write(f"Description: {description}\n")
            f.write(f"Audio File: {os.path.basename(audio_path)}\n\n")
            
            f.write("Transcript:\n")
            f.write("-" * 20 + "\n")
            # Format transcript into sentences
            sentences = transcript.replace('...', '.').split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Write formatted transcript
            for sentence in sentences:
                if sentence:
                    f.write(f"{sentence}.\n")
            f.write("\n\n")
        
        return transcript, []
    
    except Exception as e:
        print(f"\nError analyzing video: {str(e)}")
        return "", []

def process_playlist(playlist_url: str) -> None:
    """Process all videos in a YouTube playlist and generate transcripts"""
    try:
        # Configure yt-dlp options for playlist
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,  # Don't download videos, just get info
            'no_warnings': True
        }
        
        print(f"\nFetching playlist: {playlist_url}")
        print("-" * 50)
        
        # Extract playlist information
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)
            
            if not playlist_info:
                raise ValueError("Could not fetch playlist information")
            

            # Create directory for playlist outputs
            playlist_title = playlist_info.get('title', 'playlist').replace('/', '_')
            playlist_dir = os.path.join('output', playlist_title)
            os.makedirs(playlist_dir, exist_ok=True)
            
            # Process each video in the playlist
            entries = playlist_info.get('entries', [])
            total_videos = len(entries)
            
            print(f"Found {total_videos} videos in playlist: {playlist_title}")
            print("-" * 50 + "\n")
            
            for i, entry in enumerate(entries, 1):
                video_url = entry.get('url')
                video_title = entry.get('title', f'video_{i}')
                
                print(f"Processing video {i}/{total_videos}: {video_title}")
                
                try:
                    # Process video and save to playlist directory
                    transcript, _ = analyze_youtube_video(video_url, output_dir=playlist_dir)
                    
                    if transcript:
                        print("✓ Transcript generated successfully")
                    else:
                        print("⚠ No transcript generated")
                        
                except Exception as e:
                    print(f"✗ Error processing video: {str(e)}")
                
                print()
            
            print("-" * 50)
            print(f"Playlist processing complete! Results saved in: {playlist_dir}")
    
    except Exception as e:
        print(f"Error processing playlist: {str(e)}")

def is_playlist_url(url: str) -> bool:
    """Check if a URL is a playlist URL"""
    return 'playlist' in url or 'list=' in url

def main():
    # Get YouTube URL
    url = input('Enter YouTube URL or Playlist URL: ')
    
    # Check if it's a playlist or single video
    if is_playlist_url(url):
        # If URL contains both video and playlist, remove the video part
        if '&list=' in url:
            playlist_id = url.split('&list=')[1].split('&')[0]
            url = f'https://www.youtube.com/playlist?list={playlist_id}'
        process_playlist(url)
    else:
        transcript, _ = analyze_youtube_video(url)

if __name__ == '__main__':
    main()
