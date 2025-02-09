"""
Description:
    Extract text from audio/video file using WhisperX.
    
    Args:
        input_path: Path to input audio/video file
        output_path: Path to save the text file (optional, defaults to input_path with .txt extension)
        model_name: WhisperX model to use (default: large-v3-turbo)
        device: Device to use for computation (default: cuda if available, else cpu)
        batch_size: Number of parallel batches for processing (default: 4)
        compute_type: Type of computation precision (default: float16)
    
    Returns:
        The extracted text as a string

## Author Information
- **Author**: Nic Cravino
- **Email**: spidernic@me.com 
- **LinkedIn**: [Nic Cravino](https://www.linkedin.com/in/nic-cravino)
- **Date**: February 8, 2025

## License: Apache License 2.0 (Open Source)
This tool is licensed under the Apache License, Version 2.0. This is a permissive license that allows you to use, distribute, and modify the software, subject to certain conditions:

- **Freedom of Use**: Users are free to use the software for personal, academic, or commercial purposes.
- **Modification and Distribution**: You may modify and distribute the software, provided that you include a copy of the Apache 2.0 license and state any significant changes made.
- **Attribution**: Original authorship and contributions must be acknowledged when redistributing the software or modified versions of it.
- **Patent Grant**: Users are granted a license to any patents covering the software, ensuring protection from patent claims on original contributions.
- **Liability Disclaimer**: The software is provided "as is," without warranties or conditions of any kind. The authors and contributors are not liable for any damages arising from its use.

For full details, see the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
"""


import whisperx
import torch
from pathlib import Path
from typing import Union, Optional
import time

def extract_text(
    input_path: Union[str, Path], 
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "large-v3-turbo",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 4,
    compute_type: str = "float16"
) -> str:
    """
    Extract text from audio/video file using WhisperX.
    
    Args:
        input_path: Path to input audio/video file
        output_path: Path to save the text file (optional, defaults to input_path with .txt extension)
        model_name: WhisperX model to use (default: large-v3-turbo)
        device: Device to use for computation (default: cuda if available, else cpu)
        batch_size: Number of parallel batches for processing (default: 4)
        compute_type: Type of computation precision (default: float16)
    
    Returns:
        The extracted text as a string
    """
    print(f"Loading model {model_name} on {device}...")
    start_time = time.time()
    
    # Convert paths to Path objects
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('.txt')
    else:
        output_path = Path(output_path)
    
    # Load model
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        asr_options={"word_timestamps": False}  # We don't need word timestamps for text only
    )
    
    # Load audio
    print("Loading audio...")
    audio = whisperx.load_audio(str(input_path))
    
    # Transcribe with VAD
    print("Transcribing...")
    result = model.transcribe(
        audio,
        batch_size=batch_size
    )
    
    # Extract text from segments
    full_text = ""
    for segment in result["segments"]:
        full_text += segment["text"].strip() + "\n"
    
    # Save to file if output_path is provided
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    elapsed_time = time.time() - start_time
    print(f"Done! Processed in {elapsed_time:.2f} seconds")
    print(f"Text saved to: {output_path}")
    
    return full_text

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract text from audio/video files using WhisperX"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input audio/video file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output text file (optional, defaults to input path with .txt extension)"
    )
    parser.add_argument(
        "-m", "--model",
        default="large-v3-turbo",
        help="Model name (default: large-v3-turbo)"
    )
    parser.add_argument(
        "-d", "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)"
    )
    parser.add_argument(
        "-c", "--compute_type",
        default="float16",
        choices=["float16", "float32"],
        help="Computation precision type (default: float16)"
    )
    
    args = parser.parse_args()
    extract_text(
        args.input,
        args.output,
        args.model,
        args.device,
        args.batch_size,
        args.compute_type
    )
