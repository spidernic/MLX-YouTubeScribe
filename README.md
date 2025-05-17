# MLX-YouTubeScribe

> **Note**: This application uses local AI models for transcription. The models will be automatically downloaded the first time you run the application. Please ensure you have a stable internet connection for the initial setup.

A powerful application that generates transcripts from YouTube videos and playlists using local Whisper AI models for speech recognition. The application processes audio from YouTube videos, analyzes the audio characteristics, and generates accurate text transcripts completely offline after the initial model download.

## Features

- **Video & Playlist Support**: Process individual YouTube videos or entire playlists
- **High-Quality Transcription**: Utilizes OpenAI's Whisper model for accurate speech-to-text
- **Audio Analysis**: Provides detailed audio metrics including duration, sample rate, and amplitude
- **Apple Silicon Optimized**: Leverages MLX and Metal Performance Shaders (MPS) for accelerated performance on M1/M2 Macs
- **Batch Processing**: Automatically processes all videos in a playlist
- **User-Friendly Interface**: Simple web interface built with Streamlit
- **Output Formats**: Saves results in both JSON and human-readable text formats

## Prerequisites

- Python 3.8 or higher
- macOS with Apple Silicon (M1/M2) for optimal performance with MLX and PyTorch MPS
- FFmpeg (required by yt-dlp)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Transcriptor
   ```

2. Create and activate a conda environment:
   ```bash
   # Create a new conda environment with Python 3.8 or higher
   conda create -n transcriptor python=3.9
   conda activate transcriptor
   ```

3. Install the required packages:
   ```bash
   # Install PyTorch with Metal Performance Shaders (MPS) support for Apple Silicon
   conda install pytorch::pytorch torchvision torchaudio -c pytorch
   
   # Install remaining dependencies
   conda install -c conda-forge streamlit yt-dlp numpy scipy
   
   # Install MLX for Apple Silicon acceleration
   pip install mlx
   
   # Install transformers with MLX support
   pip install transformers[torch]
   ```

4. Install FFmpeg (if not already installed):
   - On macOS: `brew install ffmpeg`
   - On Ubuntu/Debian: `sudo apt install ffmpeg`
   - On Windows: Download from [FFmpeg's website](https://ffmpeg.org/download.html)

## Usage

### GUI Version
1. Run the Streamlit GUI application:
   ```bash
   streamlit run src/gui_generate_transcripts.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. Enter a YouTube video URL or playlist URL in the input field

4. Click "Generate Transcript" to start processing

5. View the transcript directly in the browser or check the `output` directory for saved files

### CLI Version

For command-line usage:
```bash
python src/cli_generate_transcripts.py [youtube_url]
```

Options:
- `youtube_url`: URL of the YouTube video or playlist to process
- The script will automatically process the video/playlist and save results to the `output` directory

## Output

The application creates an `output` directory with the following structure:

```
output/
├── [video_title].json     # Complete analysis in JSON format
├── [video_title].txt      # Human-readable transcript
└── audio/
    └── [video_title].wav  # Downloaded audio file
```

For playlists, a subdirectory with the playlist name is created containing all video transcripts.

## Models

The application uses the following models:
- `openai/whisper-large-v3-turbo` for high-quality transcription
- `openai/whisper-tiny.en` as a fallback (currently not in use)

## Performance Notes

- The application is optimized for Apple Silicon (M1/M2) using MLX for accelerated inference
- Processing time depends on video length and system performance
- For long videos, the audio is automatically split into 30-second chunks for processing

## Troubleshooting

- **FFmpeg not found**: Ensure FFmpeg is installed and added to your system PATH
- **Model download issues**: Check your internet connection and try again
- **Memory errors**: Try processing shorter videos or close other memory-intensive applications

## License

This project is open source and available under the [Apache License 2.0](LICENSE).

```
Copyright 2025 spidernic

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video downloading
- [Streamlit](https://streamlit.io/) for the web interface
