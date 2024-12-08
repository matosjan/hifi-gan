from pathlib import Path

from src.datasets.base_dataset import BaseDataset
import os
from tqdm import tqdm
import torchaudio

class TTSDataset(BaseDataset):
    def __init__(self, data_dir_path=None, text_from_cli=None, *args, **kwargs):
        data = []
        if data_dir_path == None and text_from_cli == None:
            print(f'No data given')
            exit(0)
        if text_from_cli != None:
            entry = {
                'text': text_from_cli,
                'path': f'cli_input_gen.wav',
                'audio_frames': 0
            }
            data.append(entry)
        else:
            dir_len = len(os.listdir(data_dir_path))
            for path in tqdm(Path(data_dir_path).iterdir(), total=dir_len):
                entry = {}
                entry["path"] = str(path)
                if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                    entry["audio_frames"] = self._calc_audio_frames(path)
                
                elif path.suffix in [".txt"]:
                    with path.open() as f:
                        entry["text"] = f.read().strip()
                    entry["audio_frames"] = 0
                    
                if len(entry) > 1:
                    data.append(entry)
            
                
        super().__init__(data, *args, **kwargs)

    def _calc_audio_frames(self, audio_path):
        t_info = torchaudio.info(str(audio_path))
        return t_info.num_frames

