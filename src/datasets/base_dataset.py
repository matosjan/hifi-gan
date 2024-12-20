import logging
import random

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from speechbrain.inference.TTS import Tacotron2


logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index,
        target_sr=22050,
        limit=None,
        is_train=True,
        min_audio_frames=None,
        shuffle_index=False,
        instance_transforms=None,
        frames_to_leave=None
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            target_sr (int): supported sample rate.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            max_audio_length (int): maximum allowed audio length.
            max_test_length (int): maximum allowed text length.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)
        self.is_train = is_train
        if self.is_train:
            index = self._filter_records_from_dataset(
                index, min_audio_frames
            )
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index: list[dict] = index

        self.target_sr = target_sr
        self.frames_to_leave = frames_to_leave
        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        if 'text' in data_dict.keys():
            melspec = self.get_mel_spectrogram_from_text(data_dict['text'])
            instance_data = {
                "path": data_dict['path'],
                "melspec": melspec,
            }
            

            return instance_data

        audio_path = data_dict["path"]
        audio = self.load_audio(audio_path)
        if self.is_train:
            start = random.randint(0,  audio.shape[-1] - self.frames_to_leave)
            audio = audio[:, start : start + self.frames_to_leave]

        melspec = self.get_mel_spectrogram(audio)

        instance_data = {
            "audio": audio,
            "melspec": melspec,
            "path": audio_path,
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def get_mel_spectrogram(self, audio):
        """
        Special instance transform with a special key to
        get spectrogram from audio.

        Args:
            audio (Tensor): original audio.
        Returns:
            spectrogram (Tensor): spectrogram for the audio.
        """
        return self.instance_transforms["get_mel_spectrogram"](audio)

    def get_mel_spectrogram_from_text(self, text):
        tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts", overrides={"max_decoder_steps": 10000})
        melspec, _, _ = tacotron2.encode_text(text)

        return melspec

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if transform_name == "get_mel_spectrogram":
                    continue  # skip special key
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
        min_audio_frames,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        the desired max_test_length or max_audio_length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            min_audio_length (int): minimum allowed audio length.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        initial_size = len(index)
        if min_audio_frames is not None:
            short_audios = (
                np.array([el["audio_frames"] for el in index]) < min_audio_frames
            )
            _total = short_audios.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are shorter then "
                f"{min_audio_frames} length. Excluding them."
            )
        else:
            short_audios = False

        initial_size = len(index)
        records_to_filter = short_audios

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total} ({_total / initial_size:.1%}) records  from dataset"
            )

        return index

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )   
   

    @staticmethod
    def _sort_index(index):
        """
        Sort index by audio length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["audio_frames"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
