from pathlib import Path # For file path handling (optional)
from datetime import datetime # For timestamping our output files (optional)
import json

def save_results(result: dict, audio_path: Path) -> Path:
    """
    Save transcription results to files.
    
    Args:
        result: Transcription result dictionary
        audio_path: Original audio file path
        
    Returns:
        base_path: Base path where files were saved

    Raises:
        IOError: If saving results fails
    """
    try:
        output_dir = Path("tmp/transcriptions")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = output_dir / f"{Path(audio_path).stem}_{timestamp}"
        
        # Save results in different formats
        with open(f"{base_path}.txt", 'w', encoding='utf-8') as f:
            f.write(result["text"])
        
        with open(f"{base_path}_full.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        with open(f"{base_path}_segments.json", 'w', encoding='utf-8') as f:
            json.dump(result["segments"], f, indent=2, ensure_ascii=False)

        # for next step     
        filtered = [
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", "")
            }
            for seg in result["segments"]
            ]
        with open(f"{base_path}_segments_filtered.json", 'w', encoding='utf-8') as f:
              json.dump(filtered, f, indent=2, ensure_ascii=False)
              
        
        #print(f"\nResults saved to transcriptions folder:")

        
        return base_path
    
    except IOError as e:
        print(f"Error saving results: {str(e)}")
        raise
