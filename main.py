import argparse
import asyncio
import signal
import sys
import os
import warnings
import imageio_ffmpeg
from dotenv import load_dotenv

# Suppress third-party deprecation warnings for a clean CLI output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure ffmpeg from imageio-ffmpeg is in the PATH
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

from src.di.container import configure_dependency_injection
from src.use_cases.record_meeting import RecordMeetingUseCase
from src.use_cases.transcribe_meeting import TranscribeMeetingUseCase
from src.use_cases.summarize_meeting import SummarizeMeetingUseCase
from src.use_cases.save_minutes import SaveMinutesUseCase
from src.domain.entities.meeting import AudioConfig, Meeting
from src.infrastructure.audio.wasapi_audio_capture import WasapiAudioCapture


def list_audio_devices():
    """Lists available audio devices for the user to select from."""
    capture = WasapiAudioCapture()
    devices = capture.list_devices()
    print("Available Audio Devices:\n")
    for idx, device in enumerate(devices):
        print(f"[{idx}] {device['name']} (Channels: in={device['max_input_channels']}, out={device['max_output_channels']})")
    print("\nNote: For Teams loopback on Windows, you usually don't need to specify the device if default output is used, but specifying microphone can help.")


async def main():
    parser = argparse.ArgumentParser(description="Microsoft Teams Meeting Capture & Summarizer Tool")
    parser.add_argument("--list-devices", action="store_true", help="Lists available audio devices and exits")
    parser.add_argument("--id", type=str, default="Meeting", help="The meeting identifier/name")
    parser.add_argument("--mic", type=str, default=None, help="The exact name of the microphone device to use")
    parser.add_argument("--audio-file", type=str, default=None, help="Path to an existing audio file to transcribe (skips recording)")
    
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    # 1. Load Environment Variables
    load_dotenv()

    print("Configuring Dependency Injection...")
    configure_dependency_injection()

    # 2. Instantiate Use Cases
    record_uc = RecordMeetingUseCase()
    transcribe_uc = TranscribeMeetingUseCase()
    summarize_uc = SummarizeMeetingUseCase()
    save_uc = SaveMinutesUseCase()

    if args.audio_file:
        from pathlib import Path
        from datetime import datetime
        
        audio_path = Path(args.audio_file)
        if not audio_path.exists():
            print(f"Error: Audio file not found at {audio_path}")
            sys.exit(1)
            
        print(f"Skipping recording. Using existing audio file: {audio_path}")
        # Create a dummy meeting object for the existing file
        audio_config = AudioConfig(sample_rate=44100, channels=2, dtype="int16")
        meeting = Meeting(
            id=args.id,
            started_at=datetime.now(),
            audio_config=audio_config,
            ended_at=datetime.now(),
            audio_file_path=audio_path
        )
    else:
        # Configuration for recording
        audio_config = AudioConfig(
            sample_rate=44100, 
            channels=2, # Stereo is better for mixing loopback and mic
            dtype="int16",
            microphone_device=args.mic
        )

        print(f"Starting recording for meeting: {args.id}...")
        print("Press Ctrl+C to stop recording and process the meeting.")
        
        # 3. Start Recording
        meeting = record_uc.start(args.id, audio_config)
        
        # Wait for completion (via signal)
        stop_event = asyncio.Event()

        def handle_sigint():
            print("\nCtrl+C detected! Stopping recording...")
            stop_event.set()

        # In Windows, asyncio add_signal_handler is not fully supported for SIGINT in the same way,
        # so we use loop.run_in_executor or standard signal binding
        signal.signal(signal.SIGINT, lambda sig, frame: handle_sigint())

        try:
            # Keep recording until stopped
            await stop_event.wait()
        except KeyboardInterrupt:
            pass # Handled by signal

        # 4. Stop Recording
        meeting = record_uc.stop(meeting)
        print(f"Recording saved to: {meeting.audio_file_path}")
        print(f"Meeting duration: {meeting.duration_seconds / 60:.2f} minutes")

    # 5. Transcribe
    print("\nTranscribing audio... (This may take a while depending on your hardware)")
    transcript = transcribe_uc.execute(meeting)
    print("Transcription complete.")

    # 6. Summarize
    print("\nGenerating summary with Gemini Flash...")
    summary = summarize_uc.execute(transcript)
    print("Summary generated.")

    # 7. Save Minutes
    filepath = save_uc.execute(meeting, transcript, summary)
    print(f"\nMinutes saved successfully to: {filepath}")
    print("\nDone!")

if __name__ == "__main__":
    # Windows fix for asyncio and subprocesses/ProactorEventLoop if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())