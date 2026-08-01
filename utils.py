import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

# Ensure ffmpeg is installed and accessible in your system's PATH for pydub to work correctly with various audio formats.

def audio_to_text(audio_file_path: str) -> str:
    """
    Converts an audio file (e.g., MP3, WAV) to text using the Google Web Speech API.
    This function handles longer audio files by splitting them into smaller chunks
    based on silence, processing each chunk, and then concatenating the results.

    Args:
        audio_file_path (str): The path to the audio file to be transcribed.

    Returns:
        str: The full transcribed text from the audio file. Returns an empty string
             if no speech is recognized or an error occurs.
    """
    try:
        # Load the audio file using pydub
        audio = AudioSegment.from_file(audio_file_path)
    except Exception as e:
        print(f"Error loading audio file {audio_file_path}: {e}")
        return ""

    # Split audio into chunks where silence is detected.
    # These parameters (min_silence_len, silence_thresh, keep_silence)
    # can be adjusted for better performance based on the characteristics of your audio.
    chunks = split_on_silence(audio,
                              min_silence_len=500,  # Minimum silence length in milliseconds
                              silence_thresh=-40,   # Silence threshold in dBFS (decibels relative to full scale)
                              keep_silence=200     # Keep 200ms of silence between chunks to avoid abrupt cuts
                             )

    full_text = []
    recognizer = sr.Recognizer()

    # Process each audio chunk
    for i, chunk in enumerate(chunks):
        # Export each chunk to a temporary WAV file for SpeechRecognition to process.
        chunk_filename = f"temp_audio_chunk_{i}.wav"
        try:
            chunk.export(chunk_filename, format="wav")

            with sr.AudioFile(chunk_filename) as source:
                # Adjust for ambient noise and record the audio data
                recognizer.adjust_for_ambient_noise(source)
                audio_listened = recognizer.record(source)

                try:
                    # Attempt to recognize speech using Google Web Speech API
                    text = recognizer.recognize_google(audio_listened)
                    full_text.append(text)
                except sr.UnknownValueError:
                    # Handle cases where speech is unintelligible
                    print(f"Could not understand audio in chunk {i}")
                except sr.RequestError as e:
                    # Handle API request errors (e.g., no internet connection, API limit exceeded)
                    print(f"Could not request results from Google Web Speech API service for chunk {i}; {e}")
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
        finally:
            # Ensure temporary chunk file is removed
            if os.path.exists(chunk_filename):
                os.remove(chunk_filename) 

    return " ".join(full_text)

def preprocess_text(text: str) -> str:
    """
    Performs basic text preprocessing on the input string.
    This typically includes converting text to lowercase and stripping leading/trailing whitespace.
    More advanced preprocessing steps (e.g., punctuation removal, stop word filtering,
    stemming/lemmatization) can be added here if the downstream ML model requires them.

    Args:
        text (str): The input text string to be preprocessed.

    Returns:
        str: The processed text string.
    """
    # Convert text to lowercase to ensure consistency
    # Remove leading/trailing whitespace
    return text.lower().strip()

# Example usage for direct testing of utility functions
if __name__ == "__main__":
    print("\n--- Testing utils.py directly ---")
    
    # --- Test preprocess_text function ---
    sample_text = "  This is a TEST text with some CAPITALS.   "
    processed_text = preprocess_text(sample_text)
    print(f"Original text: '{sample_text}'")
    print(f"Processed text: '{processed_text}'\n")

    # --- Test audio_to_text function (requires a valid audio file with speech) ---
    print("To test audio_to_text, please provide a WAV file with spoken words.")
    print("Example: audio_to_text('path/to/your/audio.wav')")
    # For a real test, you would have a file like 'test_speech.wav'
    # For demonstration, we'll create a dummy file but it won't be recognized by Google Speech API easily.
    # from pydub.generators import Sine
    # sine_wave = Sine(440).to_audio_segment(duration=1000)
    # silence = AudioSegment.silent(duration=500)
    # test_audio_segment = sine_wave + silence + Sine(550).to_audio_segment(duration=1000)
    # test_audio_segment.export("test_dummy_audio.wav", format="wav")
    # print("Generated test_dummy_audio.wav (note: this is a tone, not speech, so recognition will likely fail).")
    # # Uncomment to try and process the dummy audio (will likely result in UnknownValueError)
    # # dummy_audio_text = audio_to_text("test_dummy_audio.wav")
    # # print(f"Text from dummy audio: {dummy_audio_text}")
    
    # Clean up any dummy files if created by manual uncommenting
    # if os.path.exists("test_dummy_audio.wav"):
    #     os.remove("test_dummy_audio.wav")