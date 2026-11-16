import os
import sherpa_onnx
import soundfile as sf
from huggingface_hub import snapshot_download

def print_trans():
  file = "eng-test-wav/trans.txt"  
  if not os.path.exists(file):
    print(f"Please provide a valid text file at: {file}")
    return

  with open(file, "r") as f:
    print(f.read())
   

def split_wav(recognizer, audio, sample_rate):
  chunk_duration = 5 # 3 # 10 # seconds
  samples_per_chunk = sample_rate * chunk_duration
  total_samples = len(audio)
  
  # loop and process 10-sec chunks
  for i in range(0, total_samples, samples_per_chunk):
    chunk_audio = audio[i : i + samples_per_chunk]
    # create stream and decode for this chunk
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, chunk_audio)
    recognizer.decode_stream(stream)
    #print("text: ")
    print(stream.result.text)
    
  
def run_inference():
#def main():    
  # 1. Download model files from Hugging Face
  model_dir="parakeet-en-model"
  # Check available files (model.onnx or model.int8.onnx and tokens.txt)
  onnx_model = os.path.join(model_dir, "model.int8.onnx")
  if not os.path.exists(onnx_model):
    onnx_model = os.path.join(model_dir, "model.onnx")
  tokens_path = os.path.join(model_dir, "tokens.txt")

  # 2. Create OfflineRecognizer for NeMo CTC
  recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
      model=onnx_model,
      tokens=tokens_path,
      num_threads=2,
      debug=False,
      provider="cpu",
  )

  # 3. Load an audio file (must be mono, 16 kHz typically, depending on the model)
  #audio_path = "eng-test-wav/0.wav"
  audio_path = "eng-test-wav/1.wav"
  #audio_path = "eng-test-wav/8k.wav"
  #audio_path = "./eng-test-wav/Obama.wav"
  if not os.path.exists(audio_path):
    print(f"Please provide a valid audio file at: {audio_path}")
    return

  print("audio file:", audio_path)
  
  audio, sample_rate = sf.read(audio_path, dtype="float32")
  if audio.ndim > 1:
    audio = audio[:, 0]  # Take first channel if stereo

  split_wav(recognizer, audio, sample_rate)  
  

  # 4. Create stream and decode
  #stream = recognizer.create_stream()
  #stream.accept_waveform(sample_rate, audio)
  #recognizer.decode_stream(stream)
  #print("Recognized text:")
  #print(stream.result.text)
    
  print("------------ verify -------------")
  print_trans()
  