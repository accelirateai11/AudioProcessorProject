#!/usr/bin/env python3
"""
CLI tool for audio transcription service
"""
import argparse
import json
import os
import sys
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()

def transcribe_file(
    file_path: str,
    api_url: str = "http://localhost:8000",
    config: Optional[dict] = None,
    output_format: str = "text",
    verbose: bool = False
) -> dict:
    """
    Send audio file to transcription service
    
    Args:
        file_path: Path to audio file
        api_url: API base URL
        config: Optional configuration
        output_format: Output format (text/json/srt)
        verbose: Show detailed progress
    
    Returns:
        Response data
    """
    url = f"{api_url}/v1/transcribe"
    
    file_path = Path(file_path)
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        sys.exit(1)
    
    # Prepare multipart form data
    files = {"file": (file_path.name, open(file_path, "rb"))}
    
    # Add configuration if provided
    data = {}
    if config:
        data["config"] = json.dumps(config)
    
    if verbose:
        console.print(f"[bold]Sending file:[/bold] {file_path}")
        console.print(f"[bold]API URL:[/bold] {url}")
        if config:
            console.print(f"[bold]Config:[/bold] {json.dumps(config, indent=2)}")
    
    # Show progress spinner during API call
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Processing audio...[/bold green]"),
        TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("Processing", total=None)
        
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            progress.stop()
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            if hasattr(e, 'response') and e.response:
                try:
                    error_data = e.response.json()
                    console.print(f"Server response: {json.dumps(error_data, indent=2)}")
                except:
                    console.print(f"Status code: {e.response.status_code}")
                    console.print(f"Response text: {e.response.text}")
            sys.exit(1)
    
    result = response.json()
    
    return result

def format_output(result: dict, format: str = "text") -> str:
    """
    Format transcription result
    
    Args:
        result: Transcription result
        format: Output format (text, json, srt)
    
    Returns:
        Formatted output
    """
    if format == "json":
        return json.dumps(result, indent=2)
    
    if format == "text":
        return result.get("text", "")
    
    if format == "srt":
        srt_content = ""
        for i, segment in enumerate(result.get("segments", [])):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            srt_content += f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
        
        return srt_content
    
    # Default to text
    return result.get("text", "")

def format_timestamp(seconds: float) -> str:
    """Format timestamp for SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def main():
    parser = argparse.ArgumentParser(description="Audio Transcription CLI")
    
    parser.add_argument(
        "file", 
        help="Path to audio file"
    )
    
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="API URL (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--language", 
        help="Language hint (e.g., 'en')"
    )
    
    parser.add_argument(
        "--model", 
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Model size (default: small)"
    )
    
    parser.add_argument(
        "--no-separation", 
        action="store_true",
        help="Disable vocal separation"
    )
    
    parser.add_argument(
        "--diarize", 
        action="store_true",
        help="Enable speaker diarization"
    )
    
    parser.add_argument(
        "--no-vad", 
        action="store_true",
        help="Disable Voice Activity Detection"
    )
    
    parser.add_argument(
        "--format", 
        default="text",
        choices=["text", "json", "srt"],
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--output", 
        help="Output file (default: stdout)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed progress"
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        "model_size": args.model,
        "enable_separation": not args.no_separation,
        "diarize": args.diarize,
        "apply_vad": not args.no_vad
    }
    
    if args.language:
        config["language_hint"] = args.language
    
    # Process file
    try:
        result = transcribe_file(
            args.file,
            api_url=args.url,
            config=config,
            output_format=args.format,
            verbose=args.verbose
        )
        
        formatted_output = format_output(result, args.format)
        
        # Output result
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(formatted_output)
            console.print(f"[bold green]✓[/bold green] Output saved to: {args.output}")
        else:
            print(formatted_output)
            
        # Print stats in verbose mode
        if args.verbose:
            duration = result.get("duration_sec", 0)
            timings = result.get("timings_ms", {})
            
            console.print("\n[bold]Stats:[/bold]")
            console.print(f"Duration: {duration:.2f}s")
            console.print(f"Language: {result.get('language', 'unknown')}")
            console.print(f"Processing time: {timings.get('total', 0)/1000:.2f}s")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
