#!/usr/bin/env python3
"""
Transcribe videos using local Whisper model with GPU support.
"""

import argparse
import json
import logging
import unicodedata
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
import whisper
import torch

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def normalize_path(path: Path) -> Path:
    """Normalize path to NFC form to handle Vietnamese/Unicode filenames."""
    normalized_str = unicodedata.normalize("NFC", str(path))
    return Path(normalized_str)


# -----------------------------------------------------------------------------
# Transcriber class
# -----------------------------------------------------------------------------
class Transcriber:
    """Wrapper around Whisper model for video transcription."""

    def __init__(self, model_name: str = "large-v3", device: str = None):
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Whisper model '{model_name}' on {self.device}...")
        self.model = whisper.load_model(model_name, device=self.device)
        logger.info("Model loaded successfully!")

    def transcribe(self, video_path: str, language: str = None) -> Dict:
        """Transcribe a single video file and return structured transcript data."""
        video_path = normalize_path(Path(video_path))

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Transcribing: {video_path.name}")

        try:
            result = self.model.transcribe(
                str(video_path),
                language=language,
                word_timestamps=True,
                verbose=False,
            )

            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "words": segment.get("words", []),
                })

            transcript_data = {
                "text": result["text"].strip(),
                "language": result["language"],
                "segments": segments,
                "duration": result.get("duration", 0),
            }

            logger.info(
                f"✓ Transcribed {len(segments)} segments ({transcript_data['language']})"
            )
            return transcript_data

        except Exception as e:
            logger.error(f"Failed to transcribe {video_path}: {e}")
            raise


# -----------------------------------------------------------------------------
# File discovery
# -----------------------------------------------------------------------------
def find_video_files(videos_dir: Path) -> List[Path]:
    """Find all supported video files in the given directory (non-recursive)."""
    video_extensions = [".mp4", ".mkv", ".avi", ".mov", ".webm"]
    videos = []
    for ext in video_extensions:
        for video_path in videos_dir.glob(f"*{ext}"):
            videos.append(normalize_path(video_path))
    return sorted(videos)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def save_transcript(transcript_data: dict, output_path: Path) -> None:
    """Save transcript data as formatted JSON (UTF-8, Vietnamese-safe)."""
    output_path = normalize_path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Batch transcription
# -----------------------------------------------------------------------------
def transcribe_all_videos(
    videos_dir: str = "/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/input/video",
    output_dir: str = "/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/transcribe_data",
    model: str = "large-v3",
    language: str = None,
) -> None:
    """Transcribe all videos in a directory with progress tracking."""
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = find_video_files(videos_dir)

    if not video_files:
        logger.error(f"No video files found in {videos_dir}")
        return

    logger.info(f"Found {len(video_files)} video(s) in {videos_dir}")
    logger.info(f"Output directory: {output_dir}")

    transcriber = Transcriber(model_name=model)

    successful: List[str] = []
    failed: List[str] = []

    with tqdm(total=len(video_files), desc="Transcribing videos") as pbar:
        for video_path in video_files:
            output_file = normalize_path(output_dir / f"{video_path.stem}.json")

            if output_file.exists():
                logger.info(f"Skipping (already transcribed): {video_path.name}")
                successful.append(str(video_path))
                pbar.update(1)
                continue

            logger.info("=" * 80)
            logger.info(f"Transcribing: {video_path.name}")
            logger.info("=" * 80)

            try:
                transcript = transcriber.transcribe(str(video_path), language=language)
                transcript["video_file"] = video_path.name
                transcript["video_path"] = str(video_path)

                save_transcript(transcript, output_file)

                logger.info(f"✓ Saved transcript to: {output_file.name}")
                logger.info(f"  Duration: {transcript['duration']:.1f}s")
                logger.info(f"  Segments: {len(transcript['segments'])}")
                logger.info(f"  Language: {transcript['language']}")

                successful.append(str(video_path))

            except Exception as e:
                logger.error(f"Failed to transcribe {video_path.name}: {e}")
                failed.append(str(video_path))

            pbar.update(1)

    # Summary
    summary = {
        "total": len(video_files),
        "successful": len(successful),
        "failed": len(failed),
        "model": model,
        "successful_files": successful,
        "failed_files": failed,
    }
    summary_file = output_dir / "transcription_summary.json"
    save_transcript(summary, summary_file)

    separator = "=" * 80
    logger.info(separator)
    logger.info("TRANSCRIPTION SUMMARY")
    logger.info(separator)
    logger.info(f"Total videos: {summary['total']}")
    logger.info(f"Successful:   {summary['successful']}")
    logger.info(f"Failed:       {summary['failed']}")
    if failed:
        logger.warning("Failed files:")
        for f in failed:
            logger.warning(f"  - {f}")


# -----------------------------------------------------------------------------
# Single video transcription
# -----------------------------------------------------------------------------
def transcribe_single_video(
    video_path: str,
    output_dir: str = "/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/transcribe_data",
    model: str = "large-v3",
    language: str = None,
) -> None:
    """Transcribe a single video file."""
    video_path = normalize_path(Path(video_path))

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcriber = Transcriber(model_name=model)
    transcript = transcriber.transcribe(str(video_path), language=language)

    transcript["video_file"] = video_path.name
    transcript["video_path"] = str(video_path)

    output_file = normalize_path(output_dir / f"{video_path.stem}.json")
    save_transcript(transcript, output_file)

    logger.info(f"✓ Transcript saved to: {output_file}")
    logger.info(f"  Duration: {transcript['duration']:.1f}s")
    logger.info(f"  Segments: {len(transcript['segments'])}")
    logger.info(f"  Language: {transcript['language']}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CS431 Video Transcription Pipeline — Whisper-based"
    )
    parser.add_argument("--all", action="store_true", help="Transcribe all videos in --videos-dir")
    parser.add_argument("--video", help="Path to a single video file")
    parser.add_argument(
        "--videos-dir",
        default="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/input/video",
        help="Directory containing video files (default: /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/input/video)",
    )
    parser.add_argument(
        "--output",
        default="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/transcribe_data",
        help="Output directory for transcripts (default: /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/transcribe_data)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v3"],
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g. 'vi' for Vietnamese). "
             "Leave unset to let Whisper auto-detect.",
    )

    args = parser.parse_args()

    if args.all:
        transcribe_all_videos(
            videos_dir=args.videos_dir,
            output_dir=args.output,
            model=args.model,
            language=args.language,
        )
    elif args.video:
        transcribe_single_video(
            video_path=args.video,
            output_dir=args.output,
            model=args.model,
            language=args.language,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
