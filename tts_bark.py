import sys
import torch
import numpy as np
import nltk
import soundfile as sf
from transformers import AutoProcessor, BarkModel
from bark import generate_audio, SAMPLE_RATE
from bark.generation import preload_models

#print(torch.version.cuda)
#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")



# NLTK paketlerini indir
nltk.download(['punkt', 'averaged_perceptron_tagger', 'universal_tagset'])

# GPU kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"

# Bark modelini önceden yükle
preload_models()

class BarkTTS:
    def __init__(self, speaker="v2/tr_speaker_4"):
        self.speaker = speaker
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(device)
        self.model = self.model.to_bettertransformer()
        self.model.enable_cpu_offload()

    def generate_speech(self, text, output_file="output_audio.wav"):
        text = text.replace("\n", " ").strip()
        sentences = nltk.sent_tokenize(text)

        silence = np.zeros(int(0.25 * SAMPLE_RATE))  # 0.25 saniyelik boşluk
        pieces = []

        for sentence in sentences:
            audio_array = generate_audio(sentence, history_prompt=self.speaker)
            pieces += [audio_array, silence.copy()]

        sf.write(output_file, np.concatenate(pieces), SAMPLE_RATE)
        print(f"Ses dosyası kaydedildi: {output_file}")

# Komut satırından çalıştırma desteği ekleyelim
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python tts_bark.py 'Metin burada' output_audio.wav")
        sys.exit(1)

    text = sys.argv[1]  # Gelen metin
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output_audio.wav"

    tts = BarkTTS()
    tts.generate_speech(text, output_file)
