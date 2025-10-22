"""Utility functions for generation pipeline."""

import os
from pathlib import Path
from typing import Optional, List
from openai import OpenAI


def upload_file_to_openai(client: OpenAI, file_path: str) -> Optional[str]:
    """
    Upload a file to OpenAI Files API.
    
    Args:
        client: OpenAI client instance
        file_path: Path to the file (relative or absolute)
    
    Returns:
        File ID if successful, None otherwise
    """
    # Handle relative paths from assets/ folder
    if not Path(file_path).is_absolute():
        file_path = Path("assets") / file_path
    
    if not Path(file_path).exists():
        print(f"⚠️  Warning: Screenshot not found: {file_path}")
        return None
    
    if Path(file_path).is_dir():
        print(f"⚠️  Warning: Expected file but got directory: {file_path}")
        return None
    
    try:
        with open(file_path, "rb") as file_content:
            result = client.files.create(
                file=file_content,
                purpose="vision",
            )
            print(f"    ✓ Uploaded {file_path}: {result.id}")
            return result.id
    except Exception as e:
        print(f"⚠️  Warning: Failed to upload {file_path}: {e}")
        return None


def upload_screenshots(client: OpenAI, screenshots: List[str] | str) -> List[str]:
    """
    Upload multiple screenshots and return their file IDs.
    Screenshots can be individual file paths or folder paths.
    If folder path, uploads all files in the folder (non-recursive).
    
    Args:
        client: OpenAI client instance
        screenshots: List of screenshot paths or string for folder path
    
    Returns:
        List of file IDs (excludes failed uploads)
    """
    def resolve(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else Path("assets") / p

    paths: List[Path] = []

    if isinstance(screenshots, str):
        dir_path = resolve(screenshots)
        if dir_path.is_dir():
            paths = [f for f in dir_path.iterdir() if f.is_file() and not f.name.startswith(".")]
        else:
            return []
    else:
        for s in screenshots:
            p = resolve(s)
            if p.is_file():
                paths.append(p)
            elif p.is_dir():
                paths.extend(f for f in p.iterdir() if f.is_file() and not f.name.startswith("."))

    return [fid for fid in (upload_file_to_openai(client, str(p)) for p in paths) if fid]

