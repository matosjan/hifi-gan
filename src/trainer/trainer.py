from pathlib import Path

import pandas as pd

import torch
from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        gen_output = self.model(**batch)
        if gen_output['audio_pred'].shape[-1] != batch['audio'].shape[-1]:
            gen_output['audio_pred'] = F.pad(gen_output['audio_pred'], (0, batch['audio'].shape[-1] - gen_output['audio_pred'].shape[-1]))
        
        batch.update(gen_output)

        pred_melspec = self.melspec_transform(batch['audio_pred'].squeeze(1))

        # discriminator loss

        if self.is_train:
            self.optimizer_disc.zero_grad()

        mpd_pred_on_gt, _ = self.model.mpd(batch['audio'])
        mpd_pred_on_pred, _ = self.model.mpd(batch['audio_pred'].detach())

        msd_pred_on_gt, _ = self.model.msd(batch['audio'])
        msd_pred_on_pred, _ = self.model.msd(batch['audio_pred'].detach())

        batch.update({
                      'mpd_pred_on_gt_list': mpd_pred_on_gt, 
                      'mpd_pred_on_pred_list': mpd_pred_on_pred,
                      'msd_pred_on_gt_list': msd_pred_on_gt,
                      'msd_pred_on_pred_list': msd_pred_on_pred
                    })
        
        disc_loss_dict = self.criterion.disc_loss(**batch)
        if self.is_train:
            disc_loss_dict['disc_loss'].backward()
            self._clip_grad_norm(self.model.mpd)
            self._clip_grad_norm(self.model.msd)
            self.optimizer_disc.step()
            self.lr_scheduler_disc.step()
        
        batch.update(disc_loss_dict)

        # generator loss
        if self.is_train:
            self.optimizer_gen.zero_grad()

        _, mpd_gt_fmaps = self.model.mpd(batch['audio'])
        mpd_pred_on_pred, mpd_pred_fmaps = self.model.mpd(batch['audio_pred'])

        _, msd_gt_fmaps = self.model.msd(batch['audio'])
        msd_pred_on_pred, msd_pred_fmaps = self.model.msd(batch['audio_pred'])

        batch.update({
                      'mpd_pred_on_pred_list': mpd_pred_on_pred,
                      'msd_pred_on_pred_list': msd_pred_on_pred, 
                      'pred_fmap_lists': [mpd_pred_fmaps, msd_pred_fmaps], 
                      'gt_fmap_lists': [mpd_gt_fmaps, msd_gt_fmaps], 
                      'pred_melspec': pred_melspec
                    })
        
        gen_loss_dict = self.criterion.gen_loss(**batch)
        if self.is_train:
            gen_loss_dict['gen_loss'].backward()
            self._clip_grad_norm(self.model.generator)
            self.optimizer_gen.step()
            self.lr_scheduler_gen.step()

        batch.update(gen_loss_dict)
        
        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            met(**batch)
            
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
        else:
            # Log Stuff
            self.log_predictions(batch_idx, **batch)

    def log_spectrogram(self, melspec, **batch):
        mel_spectrogram_for_plot = melspec[0].detach().cpu()
        image = plot_spectrogram(mel_spectrogram_for_plot)
        self.writer.add_image("mel_spectrogram", image)
    
    def log_audio(self, audio=None, audio_pred=None, idx='', **batch):
        if audio is not None:
            self.writer.add_audio(f"gt_audio{idx}", audio[0], 22050)
        if audio_pred is not None:
            self.writer.add_audio(f"pred_audio{idx}", audio_pred[0], 22050)

    def log_predictions(self, batch_idx, audio, audio_pred, **batch):
        self.log_audio(audio=audio, audio_pred=audio_pred, idx=f'_{batch_idx}')

   

