# whisperx_mp4_text_extract

A streamlined tool to extract text from audio/video files using WhisperX 
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={INTERSPEECH 2023},
  year={2023}
}

## Features

- Fast audio/video to text transcription using WhisperX
- GPU acceleration support
- Simple command-line interface
- Outputs plain text files
- Supports batch processing
- Configurable model size and computation settings

## Requirements

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional, but recommended for faster processing)
- CUDA Toolkit (if using GPU)

## Installation

1. Clone this repository:
```bash
git clone [your-repo-url]
cd whisperx_extract
```

2. Create and activate a conda environment (recommended):
```bash
# Create conda environment
conda create -n whisperx_env python=3.8
conda activate whisperx_env

# Install PyTorch with CUDA support (adjust cuda version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge ffmpeg-python
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python extract_text.py -i "path/to/your/video.mp4"
```

This will create a text file with the same name as your input file (e.g., "video.txt").

### Advanced Options

```bash
python extract_text.py -i "input.mp4" \
                      -o "output.txt" \
                      -m "large-v3-turbo" \
                      -d "cuda" \
                      -b 4 \
                      -c "float16"
```

Parameters:
- `-i, --input`: Input audio/video file (required)
- `-o, --output`: Output text file (optional, defaults to input path with .txt extension)
- `-m, --model`: Model name (default: large-v3-turbo)
  - Options: tiny, base, small, medium, large-v3, large-v3-turbo
- `-d, --device`: Device to use (default: cuda if available, else cpu)
- `-b, --batch_size`: Batch size for processing (default: 4)
- `-c, --compute_type`: Computation precision (default: float16)
  - Options: float16, float32

## Using as a Python Module

You can also use the tool as a Python module in your own code:

```python
from extract_text import extract_text

# Basic usage
text = extract_text("video.mp4")

# Advanced usage
text = extract_text(
    input_path="video.mp4",
    output_path="custom_output.txt",
    model_name="large-v3-turbo",
    device="cuda",
    batch_size=4,
    compute_type="float16"
)
```

## Performance Notes

- GPU acceleration provides significant speed improvements
- Larger batch sizes can improve processing speed but require more memory
- Model size affects both accuracy and processing speed:
  - large-v3-turbo: Best accuracy, slower
  - medium: Good balance of accuracy and speed
  - small: Faster but less accurate
  - base/tiny: Fastest but least accurate
