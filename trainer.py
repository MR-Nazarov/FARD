# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

# from torch.autograd import Variable
import os
import os.path
import pickle
import torch
import torch.nn.functional as F
import shutil
import monai
import copy
import re
import random
import torch.nn as nn
from pytorch_msssim import ms_ssim,MS_SSIM
from losses.metrics import metrics_lpips
import lpips
from functools import partial
import dill
import losses.metrics as Metrics
import munch
import wandb
from trainers.perceptual_loss import PerceptualLoss
from collections import defaultdict, OrderedDict
import numpy as np
import torch.utils.data
import pandas as pd
import nets as Nets
import monai.metrics as Monai_metrics
import json
import inspect
import torch.utils.data.sampler
from monai.data import pad_list_data_collate
from monai.inferers import sliding_window_inference, SliceInferer
from datasets.dataloader import dataloaderClass
# import skimage.measure
from skimage.metrics import peak_signal_noise_ratio as psnr_compare
from image.my_dicom import myDicom
from tqdm import tqdm
from torchvision.utils import make_grid
import pytorch_ssim
from nets.main_model import MainModel
from pathlib import Path
import warnings
import losses as LossFunctions
from fard.output_mode import resolve as resolve_output_mode, apply as apply_output_mode

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.backends.cudnn.benchmark = True

class pytorch_trainer:

    def __init__(self, **args):
        self.ssim_loss_obj = None
        self.create_all_args(args)
        self.current_case = None  # For test loop

        # self.args = munch.Munch(args)
        # self.modelName = model
        self.reg_to_atlas = args.get('reg_to_atlas', False)
        self.pre_config()
        self.config_device()
        self.config_paths()

        # self.config_paths()
        self.config_model()
        self.config_dataset()
        self.config_logger()
        self.args.epoch = 0
        self.config_optimizer()
        #self.load_checkpoint()
        self.config_schedualer()
        self.config_metrics()
        self.config_loss()
        self.load_checkpoint()
        if self.args.test:
            self.test_dirs()
        self.running_results = {}
        # TODO: config

        # self.model = self.modelNamemodel(kernel=self.kernel, BN=bn, num_layers=num_layers)
        # self.model_noPar = self.model  # backup with no data parallel

        print('Running conf: {}'.format(self.args.conf))
        if self.args.Train:
            torch.cuda.set_device(0)  # alter with 0 and 1



        else:
            ###################3
            # torch.cuda.set_device(1)
            #######################

            #self.load_model()
            # if type_desc is None: type_desc = 'gaus sd = ' + str(self.sd)
            self.description = '2D deblur: '
            self.GPU = True
            self.HRDir = None  # for testing the stn network we also need the labels

    def config_dataset(self):
        collate_fn_name = self.args.get('collate_fn', 'collate_batch')
        if collate_fn_name == 'pad_list_data_collate':
            self.args['collate_batch'] = pad_list_data_collate
        else:
            self.args['collate_batch'] = self.collate_batch

        if torch.cuda.is_available():
            if 'workers' not in self.args:
                import multiprocessing
                suggested_workers = min(multiprocessing.cpu_count(), 8)
                self.args['workers'] = suggested_workers

        if 'persistent_workers' not in self.args and self.args.get('workers', 0) > 0:
            self.args['persistent_workers'] = True

        if 'prefetch_factor' not in self.args and self.args.get('workers', 0) > 0:
            self.args['prefetch_factor'] = 2



        self.data = dataloaderClass(self.args)

        if self.args.test and self.args.baseTargetPath is None:
            self.args.hasTarget = False
        else:
            self.args.hasTarget = True

        if self.args.test and self.args.jsonData is not None:
            if not self.args.jsonData.endswith(".pkl") and 'target' in \
                    json.load(open(self.args.jsonData, 'r'))[self.args.keyTest][0]:
                self.args.hasTarget = True
            elif self.args.jsonData.endswith(".pkl") and 'target' in \
                    pickle.load(open(self.args.jsonData, 'rb'))[self.args.keyTest][0]:
                self.args.hasTarget = True

        if self.args.imageSize is None:
            self.args.imageSize = self.data.shape[-2:]

    def config_model(self):
        if self.args.preProcessModel is not None \
                and hasattr(self, str(self.args.preProcessModel)):
            self.args.preProcessModel = getattr(self, self.args.preProcessModel)
        if self.args.postProcessModel is not None \
                and hasattr(self, str(self.args.postProcessModel)):
            self.args.postProcessModel = getattr(self, self.args.postProcessModel)
        self.model = MainModel(self.args)
        if torch.cuda.device_count() > 1 and self.args.DataParallel:
            print(f"Using {torch.cuda.device_count()} GPUs for training!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.args.device)
        #     num_of_layers = self.args.numLeyers + 1
        #     #self.layer_weights = nn.Parameter(torch.ones(num_of_layers, requires_grad=True, device=self.args.device) / num_of_layers)
        #     self.layer_weights = [
        #         nn.Parameter(torch.ones(num_of_layers, requires_grad=True, device=self.args.device) / num_of_layers),
        #         nn.Parameter(torch.ones(num_of_layers, requires_grad=True, device=self.args.device) / num_of_layers)]
        # if 'LPIPS' in self.args.metrics:
        #     self.Lpips_loss = lpips.LPIPS(net='vgg').cuda()
        # self.model.to(self.args.device) #TODO:add device and test
        if self.args.inferer:
            self.config_inference_model()

    def log_memory(self, label=""):
        """Log memory usage at key points"""
        if not torch.cuda.is_available() or self.args.dontLog:
            return

        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3

        print(f"Memory {label}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        if not self.args.dontLog:
            self.writer.log({
                f'memory/allocated_{label}': allocated,
                'memory/reserved_{label}': reserved
            }, step=self.args.epoch)

    def config_paths(self):

        project_root = Path(__file__).parent
        self.modelPath = Path(self.args.baseModelPath,
                              self.args.project_name,
                              self.args.conf).with_suffix('.pth')

        self.resultLogPath = Path(self.args.metricPath, self.args.project_name, self.args.conf).with_suffix('.pt')

        self.ckptPath = self.modelPath.with_name(self.modelPath.with_suffix('').name +
                                                 '_ckpt').with_suffix('.pth')
        self.htmlPath = Path('htmls',
                             self.args.project_name)
        self.args.jsonData = 'json_datasets/' + self.args.jsonData

        out_project = getattr(self.args, 'test_project', None) or self.args.project_name
        self.args.outputPathTest = Path(project_root,
                                        'results',
                                        out_project,
                                        self.args.conf)
        self.quantitativePath = Path('Quantitative',
                                     self.args.project_name)
        self.make_dirs()
    def config_inference_model(self):

        if self.args.inferer == 'sliding_window':
            self.infer_model = partial(
                sliding_window_inference,
                roi_size=self.args.inferenceSize,
                sw_batch_size=self.args.sw_batch_size,
                predictor=self.model,
                overlap=self.args.overlap,
                mode = 'gaussian',
                sigma_scale = 0.25
            )
        if self.args.inferer == 'slice_inferer':
            self.infer_model = SliceInferer(
                roi_size=self.args.inferenceSize,
                sw_batch_size=1,
                spatial_dim=self.args.spatial_dim,
            )
    def collate_batch(self, batch):

        # convert a list of dictionaries to a dictionaries of tensors

        if isinstance(batch[0], list):
            Dict = batch[0][0]
        elif isinstance(batch[0], dict):
            Dict = batch[0]
        else:
            Dict = batch

        dataKeys = self.args.dataKeys
        nonDatakeys = [k for k in Dict if k not in dataKeys]
        if isinstance(batch, tuple) or isinstance(batch, list):
            batchNonData = {k: [] for k in nonDatakeys}
            for b in batch:
                for k in nonDatakeys:
                    if isinstance(b, list):
                        for element in b:
                            batchNonData[k] += [element[k]]
                    else:
                        batchNonData[k] += [b[k]]
            keys = list(Dict.keys())
            ndarray = isinstance(Dict[keys[0]], np.ndarray)
            if ndarray:
                batch = {k: torch.cat([torch.from_numpy(batch[x][k][None, None])
                                       for x in range(len(batch))])
                         for k in dataKeys}
            else:
                if isinstance(batch[0], list):
                    batchList = []
                    for batch_idx in range(len(batch)):
                        batchList.append({k: torch.cat([batch[batch_idx][x][k]
                                                        for x in range(len(batch[batch_idx]))])
                                          for k in dataKeys})
                    batch = batchList
                batch = {k: torch.cat([batch[x][k]
                                       for x in range(len(batch))])
                         for k in dataKeys}
            batch = {**batch, **batchNonData}

        return munch.Munch(batch)

    def make_dirs(self):

        self.modelPath.parent.mkdir(parents=True, exist_ok=True)
        self.resultLogPath.parent.mkdir(parents=True, exist_ok=True)
        self.htmlPath.mkdir(parents=True, exist_ok=True)
        self.quantitativePath.mkdir(parents=True, exist_ok=True)
        self.args.outputPathTest.mkdir(parents=True, exist_ok=True)

    def calc_loss(self, batch):
        """Calculate perceptual loss with optional weight optimization."""
        losses = self.args.loss + ['total_loss']
        loss_dict = {k: torch.tensor(0.0, device=self.args.device) for k in losses}

        #if self.args.perceptualNet is None:
        #    return self.criterion(batch.output, batch.target)


        labels = batch.target.data.detach()

        if isinstance(batch.output, list):
            total_loss, loss_dict = self._handle_list_output(batch, labels, loss_dict, normalized_weights)
        else:
            total_loss, loss_dict = self._handle_single_output(batch, labels, loss_dict, normalized_weights)

        # Add entropy regularization only if optimization is enabled
            #entropy = self.inference[0].inference.compute_entropy(normalized_weights)
            #print("Entropy:", entropy)
            #loss_dict['entropy'] = entropy.item()
            #total_loss = total_loss - entropy #changed to plus
            #loss_dict['layer_weights'] = normalized_weights.detach().cpu().numpy()

        loss_dict['total_loss'] = total_loss.detach().cpu().numpy()
        return total_loss, loss_dict

    def _handle_single_output(self, batch, labels, loss_dict, normalized_weights=None):
        """Process single network output."""
        total_loss = torch.zeros(1, device=self.args.device, dtype=batch.input.dtype)

        if self.args.perceptualLoss:
            for i in range(batch.output.shape[1]):
                for inference in self.inference:
                    outputs = inference(batch.output)
                    label_features = inference(labels)

                    for out, lbl in zip(outputs, label_features):
                        layer_loss = self.criterion_perceptual(out, lbl.data)
                        layer_weighted_loss = layer_loss / len(outputs)
                        total_loss += self.args.loss_coeff[1] * layer_weighted_loss
                        loss_dict['vggL1'] += float(layer_weighted_loss.item())

        if self.args.SSIMloss:
            ssim_loss = (1 - self.ssim_loss_obj(batch.output, batch.target))
            total_loss += self.args.loss_coeff[2]*ssim_loss
            loss_dict["ssim"] += float(ssim_loss.item())
        if self.criterion:
            criterion_loss = self.criterion(batch.output, batch.target)
            total_loss += self.args.loss_coeff[0]*criterion_loss
            loss_dict["criterion"] = loss_dict.get("criterion", 0.0) + float(criterion_loss.item())


        return total_loss, loss_dict

    def _handle_list_output(self, batch, labels, loss_dict, normalized_weights=None):
        total_loss = torch.zeros(1, device=self.args.device, dtype=batch.input.dtype)

        if self.args.perceptualLoss:
            # Precompute all label features for each modality
            cached_label_features = {}
            for i in range(labels.shape[1]):
                label_slice = labels[:, i:i + 1, ...]
                for inf_idx, inference in enumerate(self.inference):
                    with torch.no_grad():  # Don't need gradients for labels
                        key = (i, inf_idx)
                        cached_label_features[key] = inference(x=label_slice)

            # Now use cached features in the main loop
            for i, out in enumerate(batch.output):
                for inf_idx, inference in enumerate(self.inference):
                    out_features = inference(x=out)
                    label_features = cached_label_features[(i, inf_idx)]

                    batch_loss = torch.zeros(1, device=self.args.device)
                    for out_feat, label_feat in zip(out_features, label_features):
                        layer_loss = self.criterion_perceptual(out_feat, label_feat.data) / len(out_features)
                        batch_loss += layer_loss
                    loss_dict['vggL1'] += float(batch_loss.item())
                    total_loss += self.args.loss_coeff[1] * batch_loss

            if self.args.SSIMloss:
                ssim_loss = (1 - self.ssim_loss_obj(out, labels[:, i:i + 1, ...])) / len(batch.output)
                total_loss += ssim_loss
                loss_dict["ssim"] += float(ssim_loss.item())
            del cached_label_features

        if self.criterion and not self.args.modelName == 'DnCNN':
            if self.args.modelName == "MINet" or self.args.modelName == "VANet" or self.args.modelName == "CrossCMMT":

                # Safely calculate loss with additional dtype control
                try:
                    # Try calculating each component separately with explicit type control
                    loss0 = self.criterion(batch.output[0].float(), batch.target[:, 0:1, ...].float())
                    loss1 = self.criterion(batch.output[1].float(), batch.target[:, 1:2, ...].float())

                    criterion_loss = (self.args.loss_coeff[0] * loss0 +
                                      self.args.loss_coeff[1] * loss1)
                except Exception as e:
                    print(f"Error in criterion calculation: {e}")
                    # Fallback to a safe loss value to keep training going
                    criterion_loss = torch.tensor(0.001, device=self.args.device, dtype=torch.float32)

                # Final check
                if torch.isnan(criterion_loss).any():
                    print("NaN detected in final criterion_loss!")
                    criterion_loss = torch.tensor(0.001, device=self.args.device, dtype=torch.float32)
            else:
                output_tensor = batch.output[0] if isinstance(batch.output, list) else batch.output
                criterion_loss = self.args.loss_coeff[0]*self.criterion(output_tensor, batch.target)
            total_loss += criterion_loss

        return total_loss, loss_dict
    #####################PETEL###########################

    def perceptual_loss_PETEL(self, output, target, **kwargs):

        loss = torch.empty([0]).to(output[0].device)

        for j in range(target.shape[1]):
            if self.args.Train:
                processed_output = self.inference(output[j])
            else:
                with torch.set_grad_enabled(False):
                    processed_output = self.inference(output[j])
            with torch.set_grad_enabled(False):
                processed_target = self.inference(target[:, j:j + 1])

            for idx in range(len(processed_output)):
                loss_ = self.L1Loss(processed_output[idx], processed_target[idx], **kwargs)
                loss = torch.cat((loss, loss_[None]))

        return loss.mean()

    def profile_training(self, batch, detailed=False):
        """Profile both memory usage and execution time of training steps"""
        import time
        import torch.autograd.profiler as profiler

        print("=== Starting Training Step Profiling ===")

        # Memory before any operations
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"Initial CUDA memory: {mem_before:.1f} MB")

        # Move data to device and check memory
        start = time.time()
        batch = self.move_data_to_device(batch)
        mem_after_move = torch.cuda.memory_allocated() / 1024 ** 2
        print(
            f"After moving data: {mem_after_move:.1f} MB (+{mem_after_move - mem_before:.1f} MB) in {time.time() - start:.3f}s")

        # Detailed profiling of forward pass
        start = time.time()
        with profiler.profile(use_cuda=True, profile_memory=True) as prof:
            with profiler.record_function("model_forward"):
                batch.output = self.model(batch.input)

        mem_after_forward = torch.cuda.memory_allocated() / 1024 ** 2
        print(
            f"After forward pass: {mem_after_forward:.1f} MB (+{mem_after_forward - mem_after_move:.1f} MB) in {time.time() - start:.3f}s")

        # Loss calculation
        start = time.time()
        with profiler.profile(use_cuda=True, profile_memory=True) as prof_loss:
            with profiler.record_function("loss_calculation"):
                loss, loss_dict = self.calc_loss(batch)

        mem_after_loss = torch.cuda.memory_allocated() / 1024 ** 2
        print(
            f"After loss calculation: {mem_after_loss:.1f} MB (+{mem_after_loss - mem_after_forward:.1f} MB) in {time.time() - start:.3f}s")

        # Backward pass
        start = time.time()
        loss.backward()
        mem_after_backward = torch.cuda.memory_allocated() / 1024 ** 2
        print(
            f"After backward pass: {mem_after_backward:.1f} MB (+{mem_after_backward - mem_after_loss:.1f} MB) in {time.time() - start:.3f}s")

        # Optimizer step
        start = time.time()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        mem_after_optim = torch.cuda.memory_allocated() / 1024 ** 2
        print(
            f"After optimizer step: {mem_after_optim:.1f} MB (+{mem_after_optim - mem_after_backward:.1f} MB) in {time.time() - start:.3f}s")

        # Print detailed profile info if requested
        if detailed:
            print("\nDetailed Forward Pass Profile:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            print("\nDetailed Loss Calculation Profile:")
            print(prof_loss.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print("=== Profiling Complete ===")

        # Clear memory
        del batch.output
        torch.cuda.empty_cache()

        return {
            "memory": {
                "initial": mem_before,
                "after_data_move": mem_after_move,
                "after_forward": mem_after_forward,
                "after_loss": mem_after_loss,
                "after_backward": mem_after_backward,
                "after_optim": mem_after_optim
            }
        }

    def train_step(self, batch, loader):
        """Optimized training step with gradient accumulation and mixed precision"""
        batch = self.move_data_to_device(batch)

        # Clear gradients at the start of each step
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        if self.args.amp:
            with torch.cuda.amp.autocast():
                # Compute forward pass (always use plain model during training)
                batch.output = self.model(batch.input)

                # # Clamp output for BRATS
                # if self.args.project_name == 'BRATS':
                #     if isinstance(batch.output, list):
                #         batch.output = [torch.clamp(out, 0, 1) for out in batch.output]
                #     else:
                #         batch.output = torch.clamp(batch.output, 0, 1)

                loss, loss_dict = self.calc_loss(batch)
                    # You can add more detailed inspection here
            # Scale loss and compute gradients
            self.scaler.scale(loss).backward()

            # Gradient clipping if needed
            if self.args.gradient_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            # Standard precision training (always use plain model during training)
            batch.output = self.model(batch.input)

            # Clamp output for BRATS
            # if self.args.project_name == 'BRATS':
            #     if isinstance(batch.output, list):
            #         batch.output = [torch.clamp(out, 0, 1) for out in batch.output]
            #     else:
            #         batch.output = torch.clamp(batch.output, 0, 1)

            loss, loss_dict = self.calc_loss(batch)

            if self.train_round:
                batch.output = self.process_output(batch.output)

            # Backward pass
            loss.backward()

            # Gradient clipping if needed
            if self.args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            self.optimizer.step()

        # Update metrics
        batch_size = batch.input.size(0)
        self.running_results["loss"] += loss.item() * batch_size
        self.running_results['batch_sizes'] += batch_size

        # Update progress bar
        avg_loss = self.running_results['loss'] / self.running_results['batch_sizes']
        loader.set_description(
            f'[{self.args.epoch}/{self.args.numEpochs}] Loss: {avg_loss:.8f}'
        )

        self.loss_dict = loss_dict

        # Clear memory
        torch.cuda.empty_cache()


    def process_output(self,output):
        if isinstance(output, list):
            return [torch.round(out) for out in output]
        return torch.round(output)


    # train
    def train(self):
        for epoch in range(self.args.epoch, self.args.numEpochs + 1):
            self.args.epoch = epoch
            self.log_memory("epoch_start")
            self.model.train()

            # Initialize metrics tracking
            loader = tqdm(self.data.train, position=0, leave=True)
            self.running_results = {'batch_sizes': 0, 'loss': 0}

            # Get gradient accumulation settings from args
            use_gradient_accumulation = self.args.get('use_gradient_accumulation', False)
            accumulation_steps = self.args.get('accumulation_steps', 1) if use_gradient_accumulation else 1

            # Reset gradients at the beginning
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(loader):
                self.t = batch_idx
                batch = munch.Munch(batch)
                self.data.train.dataset.update_indices()

                if use_gradient_accumulation:
                    # With gradient accumulation
                    loss = self.train_step_with_accumulation(
                        batch=batch,
                        loader=loader,
                        accumulation_steps=accumulation_steps
                    )

                    # Only update weights after accumulating for specified steps
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(loader)):
                        if self.args.gradient_clip:
                            if self.args.amp:
                                self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                        if self.args.amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    # Standard training (one batch at a time)
                    self.train_step(batch=batch, loader=loader)

                self.process_output_batch(batch)


            # Save checkpoint if needed
            if self.args.epochToSave and epoch % self.args.epochToSave == 0:
                self.save_checkpoint()

            # Log information and update learning rate
            self.log_information(training=True)
            self.update_scheduler()

            # Validation step
            if self.args.val:
                self.eval()
            else:
                self.process_output_train()
            # Reset dataset indices
            self.data.train.dataset.init_epoch_indices()
            self.log_memory("epoch_start")

        # Save final model
        torch.save(self.model.state_dict(), self.modelPath)
        print(f'Saving final model to: {self.modelPath}')
        if not self.args.dontLog and self.args.logger == 'wandb':
            wandb.finish()

    def train_step_with_accumulation(self, batch, loader, accumulation_steps=1):
        """Optimized training step with gradient accumulation"""
        batch = self.move_data_to_device(batch)

        if self.args.amp:
            with torch.cuda.amp.autocast():
                # Compute forward pass (always use plain model during training)
                batch.output = self.model(batch.input)

                # Clamp output for BRATS
                # if self.args.project_name == 'BRATS':
                #     if isinstance(batch.output, list):
                #         batch.output = [torch.clamp(out, 0, 1) for out in batch.output]
                #     else:
                #         batch.output = torch.clamp(batch.output, 0, 1)

                loss, loss_dict = self.calc_loss(batch)

                # Scale loss by accumulation steps
                scaled_loss = loss / accumulation_steps

            # Scale loss and compute gradients (accumulate)
            self.scaler.scale(scaled_loss).backward()

        else:
            # Standard precision training (always use plain model during training)
            batch.output = self.model(batch.input)

            # Clamp output for BRATS
            # if self.args.project_name == 'BRATS':
            #     if isinstance(batch.output, list):
            #         batch.output = [torch.clamp(out, 0, 1) for out in batch.output]
            #     else:
            #         batch.output = torch.clamp(batch.output, 0, 1)

            loss, loss_dict = self.calc_loss(batch)

            # Scale loss for accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

        # Keep the original output for processing
        if self.train_round:
            batch.output = self.process_output(batch.output)

        # Update metrics with the full (unscaled) loss
        batch_size = batch.input.size(0)
        self.running_results["loss"] += loss.item() * batch_size
        self.running_results['batch_sizes'] += batch_size

        # Update progress bar
        avg_loss = self.running_results['loss'] / self.running_results['batch_sizes']
        loader.set_description(
            f'[{self.args.epoch}/{self.args.numEpochs}] Loss: {avg_loss:.8f}'
        )

        self.loss_dict = loss_dict
        del scaled_loss
        torch.cuda.empty_cache()
        return loss

    def run_model_on_patches(self, batch, patch_size=64):
        """
        use this function when there is a need to divide the image to patches in order to test the model
        this is due to memory limitations and occurs usually in pyramid attentions nets
        """
        rows, cols = batch.shape[2], batch.shape[3]
        # go over the cols to generate the large image
        for row in range(0, rows, patch_size):
            for col in range(0, cols, patch_size):
                if col == 0:
                    patch_Output = torch.round(self.model(batch[..., row:row + patch_size, col:col + patch_size]))
                else:
                    patch_Output = torch.cat(
                        (patch_Output, torch.round(self.model(batch[..., row:row + patch_size, col:col + patch_size]))),
                        dim=3)
            if row == 0:
                Output = patch_Output
            else:
                Output = torch.cat((Output, patch_Output), dim=2)

        return Output

    def iterate_series(self, description, Enlarge=True, resiz=1.414,
                       iters=2):  # resiz=1.26, iters=3 resiz=1.414, iters=2
        self.DCM = DCM_preprocessing()
        OrigDir = self.OrigDir
        for i in range(1, iters + 1):
            print("We're on iteration %d" % (i))
            # # bicubic enlargement
            if Enlarge:
                print('Resizing series by factor of ' + str(resiz))
                EnlargedDir = OrigDir + '_En(' + str(resiz) + ')X%d' % i
                description += ' En%d' % i
                if os.path.isdir(EnlargedDir):
                    print('Enlarged series already exists in directory: ' + EnlargedDir)
                else:
                    self.DCM.resize_series(resiz, OrigDir, EnlargedDir)
            else:
                EnlargedDir = OrigDir

            # CNN deblur
            print('Running series through deblurring CNN #' + str(i))

            # path for saving
            HR_path = EnlargedDir + '_D%d' % i
            OrigDir = HR_path
            description += ' D%d' % i
            if os.path.isdir(HR_path):
                print('Deblurred series already exists in directory: ' + HR_path)
            else:
                self.run(HR_path, EnlargedDir, description)
        self.HR_path = HR_path


    def config_optimizer(self):
        # Number of optimization steps
        self.optimizer = getattr(torch.optim, self.args.optimizationMethod)
        inputArgs = inspect.getfullargspec(self.optimizer.__init__).args
        optimArgs = {k: self.args[k] for k in self.args if k in inputArgs}

        # Collect all parameters to optimize
        params = list(self.model.parameters())

        # Create optimizer with all parameters
        self.optimizer = self.optimizer(params,
                                        lr=self.args.learningRate,
                                        **optimArgs)

    def config_schedualer(self):

        if self.args.cosineAnneal:
            #was self.args.T_0
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,
                                                                                  T_0=self.args.T_0,
                                                                                  eta_min=self.args.etaMin,
                                                                                  T_mult=self.args.T_mult)


    def load_pretrained(self, config):
        module = 'SWIN_UNETR'
        module = getattr(Nets, module)
        # configPath = Path(self.args.perceptualNetConfig)
        # config = yaml.load(configPath.open('r'),
        #                   Loader=CLoader)
        if hasattr(module, 'load_pretrained'):
            module.load_pretrained(config=config,
                                   model=self.perceptualNet)

    def create_loss(self, loss_name, loss_module=None):
        """Create loss instance with relevant args from Munch config."""
        # Get the loss class
        if loss_module and hasattr(loss_module, loss_name):
            loss_class = getattr(loss_module, loss_name)
        else:
            loss_class = getattr(torch.nn, loss_name)

        # Extract relevant args and create instance
        sig = inspect.signature(loss_class.__init__)
        kwargs = {k: getattr(self.args, k) for k in sig.parameters
                  if k != 'self' and hasattr(self.args, k)}

        return loss_class(**kwargs)

    def config_loss(self):
        # Main criterion
        self.criterion = self.create_loss(self.args.losses, LossFunctions)

        # Perceptual loss
        if self.args.perceptualLoss:
            self.criterion_perceptual = self.create_loss(self.args.criterion, LossFunctions)

            self.inference = []
            for index, pl in enumerate(self.args.perceptualLoss.split(',')):
                self.inference.append(PerceptualLoss(self.args, index))
                if torch.cuda.is_available():
                    self.inference[index].cuda()

        # SSIM loss
        if self.args.SSIMloss:
            #self.ssim_loss_obj = MS_SSIM() if self.args.loss2D else pytorch_ssim.SSIM3D()
            self.ssim_loss_obj = pytorch_ssim.SSIM() if self.args.loss2D else pytorch_ssim.SSIM3D()
            if torch.cuda.is_available():
                self.ssim_loss_obj.cuda()
    def _loggable_config(self):
        """The run config with credential-shaped keys stripped.

        wandb uploads whatever it is given as run config, so anything secret in
        the conf would end up in the run's config panel and in any shared link.
        """
        return {k: v for k, v in self.args.items()
                if not self.SECRET_ARG_PATTERN.search(str(k))}

    def config_logger(self):
        # dontLog means offline: no run is created at all.
        if self.args.dontLog:
            wandb.init(mode='disabled')
        else:
            wandb.init(project=self.args.project_name, name=self.args.get('conf'),
                       config=self._loggable_config())
        if not self.args.hasTarget:
            return

        project_name = self.args.project_name
        if self.args.test:
            project_name += '_test'
        if not self.args.dontLog:
            self.writer = wandb
        self.excelPath = self.quantitativePath.joinpath(self.args.conf).with_suffix('.xlsx')
        if self.args.test:
            self.excelCols = defaultdict(list)
            self.excelRowNames = []

    def create_all_args(self, kwargs):

        self.args = munch.munchify(kwargs)
        inputArgs = inspect.getfullargspec(self.__init__).args
        _locals = locals()
        self.allArgs = {x: _locals[x] for x in inputArgs if x != 'self'}
        self.allArgs = {**self.allArgs, **kwargs}

    def config_metrics(self):
        self.bestEpoch = False
        self.bestEpochMetric = 0
        self.metricStats = {'Train': defaultdict(lambda: defaultdict(list)),
                            'Validation': defaultdict(lambda: defaultdict(list))}
        self.metricFuncs = {}
        for metric in self.args.metrics:
            if 'monai' not in metric:
                if metric == 'LPIPS':
                    metrics_lpips.initialize_lpips()  # Initialize LPIPS model
                self.metricFuncs[metric] = getattr(Metrics, metric)

    def log_information(self, training):
        """
        Logs training or validation metrics and perceptual weights.

        Parameters:
        training (bool): Indicates whether the current mode is training or validation.

        Returns:
        None

        The function logs learning rates if training and a scheduler is available.
        It also logs mean and standard deviation of metrics for either the 'Train' or 'Validation' group,
        depending on the mode. If test mode is enabled, it appends the logged metrics to Excel columns
        for later export and adds the directory name of the dataset being evaluated to the Excel row names.
        If perceptual optimization is enabled, perceptual weights are logged and added to Excel columns during test mode.
        """

        if self.args.dontLog:
            return None
        if not self.args.hasTarget:
            return None
        if training and hasattr(self, 'scheduler'):
            self.log_lr()
        group = 'Train' if training else 'Validation'
        lossToTake = 'epochLoss' if training else 'evalLoss'

        # Log perceptual weights
        #     weights = self.inference[0].inference.layer_weights.detach().cpu()
        #     normalized_weights = F.softmax(weights + 1e-8, dim=0)
        #
        #     # Log each weight
        #     for i, w in enumerate(normalized_weights):
        #         weight_name = f'{group}/perceptual_weight_{i}'
        #         self.writer.log({weight_name: w.item()}, step=self.args.epoch)

            # If in test mode, add weights to excel
            # if self.args.test:
            #         normalized_weights.tolist()
            #     )
        if hasattr(self, 'loss_dict'):
            for loss_name, loss_value in self.loss_dict.items():
                self.writer.log({f'{group}/losses/{loss_name}': loss_value}, step=self.args.epoch)
        metrics = self.metricStats[group][self.args.epoch]
        for k in metrics:
            if isinstance(metrics[k][0], torch.Tensor):
                metrics_list = [x.float() for x in metrics[k]]  # Ensure float type for calculations
            else:
                metrics_list = metrics[k]
            if not self.args.test_save:
                metrics_list = metrics_list[self.args.cut_slice[0]:-self.args.cut_slice[1]]
            metric = torch.tensor(metrics_list)
            valid_metric = metric[torch.isfinite(metric)]

            meanMetric = valid_metric.mean()
            stdMetric = valid_metric.std()

            # Log mean
            self.writer.log({f'{group}/{k}/mean': meanMetric}, step=self.args.epoch)
            # Log standard deviation
            self.writer.log({f'{group}/{k}/std': stdMetric}, step=self.args.epoch)

            if self.args.test:
                if k not in self.excelCols:
                    self.excelCols[k] = {'mean': [], 'std': []}
                self.excelCols[k]['mean'].append(meanMetric.item())
                self.excelCols[k]['std'].append(stdMetric.item())
                if self.args.jsonData is None:
                    self.excelRowNames.append(self.data.evalDataset.args.inputDirs[
                                                  self.data.evalDataset.inputDirIdx - 1].name)

    def log_lr(self):

        lrEpoch = self.scheduler.get_last_lr()[0]
        if self.args.dontLog:
            return
        self.writer.log({f'Learning-Rate': lrEpoch},step=self.args.epoch)

    def calc_metrics(self, batch, training=True):
        """

        Each derived class should implement this method
        :param batch: a single batch of training\validation, is a dictionary with the following keys:
        ['input', 'target', 'output']

        :param training: if True then training
        Computes the metricStats arguments

        the metricStats is a dictionary of the following form:
        group\epoch\metric
        For instance:
        metricStats = {'Train':
                                {1:
                                    {'precision': [0.5, 0.2, 0.3],
                                    'Recall': [0.8, 0.4, 0.7]},
                                2:
                                    {'precision': [0.1, 0.3, 0.7],
                                    'Recall': [0.7, 0.7, 0.9]}}
                        'Validation':
                                {1:
                                    {'precision': 0.3,
                                    'Recall': 0.6},
                                2:
                                    {'precision': 0.1,
                                    'Recall': 0.5}}
                            }
        """

        # if not self.args.hasTarget:
        #     return

        group = 'Train' if training else 'Validation'
        batch['training'] = training
        x = self.pre_metric_calculation(batch)
        if x is not None and \
                (('confMat' in x and x['confMat'].sum() > 0)
                 or 'confMat' not in x):
            for metric in self.args.metrics:
                if not self.args.separate_metrics:
                    if 'monai' in metric:
                        real_metric = metric.replace('monai_', '')
                        _m = Metrics.monai.metrics.compute_confusion_matrix_metric(confusion_matrix=x['confMat'],
                                                                                   metric_name=real_metric)
                        if _m[0].isnan().sum() == 0:
                            _m = _m[x['batch'].target[:, 0]]
                        _m = _m[~torch.isnan(_m)]
                        _m = _m.to('cpu').detach()
                    else:
                        _m = self.metricFuncs[metric](x).to('cpu').detach()
                    _m = self.post_metric_calculation(_m, **x)
                    metric_name = metric \
                        if 'monai' not in metric \
                        else real_metric
                    self.metricStats[group][self.args.epoch][metric_name].append(_m)
                else:
                    for i in range(x.output.shape[1]):
                        x1 = copy.deepcopy(x)
                        x1.output = x1.output[:, i:i + 1, ...]
                        x1.target = x1.target[:,i:i + 1, ...]
                        if 'monai' in metric:
                            real_metric = metric.replace('monai_', '')
                            _m = Metrics.monai.metrics.compute_confusion_matrix_metric(confusion_matrix=x['confMat'],
                                                                                       metric_name=real_metric)
                            if _m[0].isnan().sum() == 0:
                                _m = _m[x['batch'].target[:, 0]]
                            _m = _m[~torch.isnan(_m)]
                            _m = _m.to('cpu').detach()
                        else:
                            _m = self.metricFuncs[metric](x1).to('cpu').detach()
                        _m = self.post_metric_calculation(_m, **x1)
                        metric_name = metric \
                            if 'monai' not in metric \
                            else real_metric
                        self.metricStats[group][self.args.epoch][metric_name + ' ' + self.args.mod[i]].append(_m)




    def post_metric_calculation(self, metric, **kwargs):
        """

        In case we have multiple classes we want to output only one of the metrics corresponding to one of the classes
        We need to tell which one of the classes is the one that we want to look at.
        Or we can do any other post-processing operation
        :param metric: Output batches
        :return: The post processed metric
        """

        return metric

    def pre_metric_calculation(self, outputBatches, training=True):
        """

        A method that must be implemented, takes the outputBatchs dict, and creates the correct
        input for calculation of the metrics for the specific problem
        :param outputBatches: Output batches
        :return: a dict with the kwargs that match the metrics that are used
        """
        if not training:
            batch_all = defaultdict()
            batch_all['all'] = outputBatches
            for i in range(outputBatches.output.shape[1]):
                temp_dict = copy.deepcopy(outputBatches)
                temp_dict.output = outputBatches.output[:, i:i + 1, ...]
                temp_dict.target = outputBatches.target[:, i:i + 1, ...]
                batch_all[str(i)] = temp_dict
        else:
            batch_all = outputBatches

        return batch_all

    def process_output_batch(self, batch):
        if type(batch.output) is list:
            if self.args.cnnModule == 'FSMNet':
                batch.output = batch.output[0:1]
            batch.output = self.convert_list_to_tensor(batch.output)
        self.calc_metrics(batch)

    def convert_list_to_tensor(self, output):
        processed_output = torch.cat(output, dim=1)
        return processed_output

    def config_device(self):
        """
         is designed to configure the computing device for your Python program, particularly in the context of PyTorch
        """
        if torch.cuda.is_available() and self.args.device != 'cpu':
            self.args.device = torch.device("cuda:0")
            torch.cuda.set_device(self.args.device)
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=2**8,
                growth_factor=1.5,
                backoff_factor=0.5,
                growth_interval=100
            ) if self.args.amp else None

            torch.cuda.empty_cache()
            #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        else:
            self.args.device = torch.device("cpu")
        if self.args.seed is not None:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print("Ablation study mode: deterministic but slower")
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print("Training mode: faster but minor variations possible")

    def save_checkpoint(self):
        state = {
            'epoch': self.args.epoch,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict(),
            'bestMetric': self.bestEpochMetric,
        }
        if self.args.cosineAnneal:
            state['scheduler'] = self.scheduler.state_dict()

        print(f'saving checkpoint for epoch {self.args.epoch}')
        torch.save(state, self.ckptPath.as_posix())

    def update_scheduler(self):
        if hasattr(self, 'scheduler'):
            self.scheduler.step()

    def set_all_seeds(self):
        """Set all random seeds for reproducibility"""
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        # pytorch
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        os.environ['PYTHONHASHSEED'] = str(self.args.seed)
        print(f"All random seeds set to: {self.args.seed}")

    def pre_config(self):
        # Output post-processing is conf data, not a project-name string match.
        self.output_mode, self.train_round = resolve_output_mode(self.args)
        if self.args.seed != 'none':
            self.set_all_seeds()
        self.remove_tmp_dirs()

    def remove_tmp_dirs(self):
        tmpJPGS = Path('/tmp/jpgs')
        if tmpJPGS.exists():
            shutil.rmtree(tmpJPGS)

    def move_data_to_device(self, batch) -> munch.Munch:
        """Optimized data transfer to device"""
        if not isinstance(batch, munch.Munch):
            batch = munch.Munch(batch)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.device != self.args.device:
                batch[k] = v.to(self.args.device, non_blocking=True)  # Enable async transfer

        return batch


    def eval(self):
        self.model.eval()
        loader = tqdm(self.data.eval, position=0, leave=True)
        keys = self.args.mod
        total_batch = len(loader)
        middle_batch_idx = total_batch // 2

        catOutputs = self.initialize_outputs()

        self.args.evalLoss = defaultdict(list)
        for t, batch in enumerate(loader):
            self.t = t
            batch = munch.Munch(batch)
            loader.set_description(f"evaluation for Epoch :"
                                   f"{self.args.epoch}")
            self.data.eval.dataset.update_indices()
            self.eval_step(batch=batch)
            if t == middle_batch_idx and self.args.get('save_img', True) and not self.args.dontLog and not self.args.test:
                output_for_save = batch.output[0] if isinstance(batch.output, list) else batch.output
                self.save_images(output_for_save.detach().cpu(), batch.input, keys)

            catOutputs = self.process_output_batch_eval(batch, catOutputs)  # TODO: change to concat batches
        self.process_output_eval()
        self.log_information(training=False)
        if not self.args.test:
            self.data.eval.dataset.init_epoch_indices()
            self.model.train()
        elif not self.args.jsonData.endswith('pkl'):
            self.dicom_results(catOutputs)
        else:
            self.nifti_results(catOutputs)

    def save_images(self, output_images, input_images, keys):
        """
        Save middle slice of input and output images to wandb for each modality.
        Works with both 3D tensors [B, H, W] and 4D tensors [B, C, H, W].

        Args:
            output_images (torch.Tensor): Output images from model [B, H, W] or [B, C, H, W]
            input_images (torch.Tensor): Input images [B, H, W] or [B, C, H, W]
            keys (list): List of modality keys
        """
        for i, key in enumerate(keys):
            ch = min(i, output_images.shape[1] - 1) if output_images.ndim >= 4 else 0
            # Handle different tensor dimensions
            if output_images.ndim == 5:  # [B, C, H, W, D] volumetric
                mid = output_images.shape[-1] // 2
                output_image = output_images[:, ch, :, :, mid]  # [B, H, W]
            elif output_images.ndim == 3:  # [B, H, W]
                output_image = output_images
            else:  # [B, C, H, W]
                output_image = output_images[:, ch, :, :]

            in_ch = min(i, input_images.shape[1] - 1) if input_images.ndim >= 4 else 0
            if input_images.ndim == 5:  # [B, C, H, W, D] volumetric
                mid = input_images.shape[-1] // 2
                input_image = input_images[:, in_ch, :, :, mid]  # [B, H, W]
            elif input_images.ndim == 3:  # [B, H, W]
                input_image = input_images
            else:  # [B, C, H, W]
                input_image = input_images[:, in_ch, :, :]

            # Add channel dimension if it's missing
            if len(output_image.shape) == 3:  # [B, H, W]
                output_image = output_image.unsqueeze(1)  # [B, 1, H, W]

            if len(input_image.shape) == 3:  # [B, H, W]
                input_image = input_image.unsqueeze(1)  # [B, 1, H, W]

            # Take only batch_size number of images
            output_image = output_image[:self.args.batchSize]
            input_image = input_image[:self.args.batchSize]

            # Create image grids
            from torchvision.utils import make_grid
            output_grid = make_grid(output_image, nrow=2, padding=2, normalize=False)
            input_grid = make_grid(input_image, nrow=2, padding=2, normalize=False)

            # Convert to numpy for wandb
            output_grid_np = output_grid.cpu().numpy().transpose(1, 2, 0)
            input_grid_np = input_grid.cpu().numpy().transpose(1, 2, 0)

            # Log both input and output to wandb
            self.writer.log({
                f'eval/{key}_input_middle_batch': self.writer.Image(
                    input_grid_np,
                    caption=f"{key} Input (Epoch {self.args.epoch})"
                ),
                f'eval/{key}_output_middle_batch': self.writer.Image(
                    output_grid_np,
                    caption=f"{key} Output (Epoch {self.args.epoch})"
                )
            }, step=self.args.epoch)



    def eval_step(self, batch):
        self.args.training = False
        batch = self.move_data_to_device(batch)

        #if isinstance(batch.input, torch.Tensor) and hasattr(batch.input, 'meta'):
        #    if 'spatial_shape' in batch.input.meta:
        #        original_spatial_shape = batch.input.meta['spatial_shape']
        #        print(f"Found original spatial shape: {original_spatial_shape}")
        with torch.no_grad():
            if self.args.inferer == 'sliding_window':
                # If input is batched, show progress for each item in batch
                if isinstance(batch.input, list):
                    outputs = []
                    for input_item in tqdm(batch.input, desc="Processing batch items"):
                        output = self.infer_model(input_item)
                        outputs.append(output)
                    batch.output = outputs
                else:
                    batch.output = self.infer_model(batch.input)
                    if isinstance(batch.output, tuple):
                        batch.output = batch.output[0]

                #print(f"Output shape: {batch.output.shape}")

                # Clamp output for BRATS / motion_corrupt (all data is [0,1] normalised)
                # unless no_clamp is set (to reproduce legacy unclamped outputs)
                batch.output = apply_output_mode(batch.output, self.output_mode)
            else:
                # For standard inference
                batch.output = self.model(batch.input)
                if isinstance(batch.output, tuple):
                    batch.output = batch.output[0]

                # Clamp output for BRATS / motion_corrupt (all data is [0,1] normalised)
                # unless no_clamp is set (to reproduce legacy unclamped outputs)
                batch.output = apply_output_mode(batch.output, self.output_mode)
        # Apply inverse transforms if enabled
        if hasattr(self.args, 'apply_inverse_transforms') and self.args.apply_inverse_transforms:
            batch = self.apply_inverse_transforms(batch)

    def apply_inverse_transforms(self, batch):
        """Apply inverse transforms dynamically based on config"""
        from monai.data import MetaTensor
        from monai.transforms import ResizeWithPadOrCrop
        import torch

        try:
            # Convert list output to tensor if needed (take first element)
            if isinstance(batch.output, list):
                batch.output = batch.output[0]

            # Get original shape from input metadata
            if isinstance(batch.input, MetaTensor) and hasattr(batch.input, 'meta'):
                original_shape = batch.input.meta.get('spatial_shape', None)

                if original_shape is None:
                    print("Warning: Could not find original_shape in metadata")
                    return batch

                # Convert tensor to list if needed
                if isinstance(original_shape, torch.Tensor):
                    if original_shape.dim() == 2:
                        original_shape = original_shape[0].cpu().tolist()
                    else:
                        original_shape = original_shape.cpu().tolist()

                # Get transforms from config
                transform_config = self.args.get('transforms', {})

                if not transform_config:
                    print("Warning: No transforms found in config, returning output as-is")
                    return batch

                # Build inverse transform chain by reversing the forward transforms
                inverse_ops = []
                # Track shape with channel dimension for proper indexing
                # Assume 1 channel, then add spatial dimensions
                current_shape_with_channel = [1] + list(original_shape)

                # Parse forward transforms and track shape changes
                for transform_name, transform_params in transform_config.items():
                    if transform_name == 'Transposed':
                        indices = transform_params.get('indices', [0, 1, 2])
                        current_shape_with_channel = [current_shape_with_channel[i] for i in indices]
                        inverse_ops.append(('transpose', indices))

                    elif transform_name == 'Flipd':
                        spatial_axis = transform_params.get('spatial_axis', 0)
                        inverse_ops.append(('flip', spatial_axis))

                    elif transform_name in ['ResizeWithPadOrCropd', 'Resized']:
                        target_size = transform_params.get('spatial_size', None)
                        if target_size:
                            spatial_shape = current_shape_with_channel[1:]
                            inverse_ops.append(('resize', spatial_shape.copy()))
                            current_shape_with_channel = [current_shape_with_channel[0]] + list(target_size)

                    elif transform_name == 'DivisiblePadd':
                        # DivisiblePadd symmetrically pads each spatial dim up to a
                        # multiple of k; inverse is a center-crop back to original shape.
                        k = transform_params.get('k', 1)
                        spatial_shape = current_shape_with_channel[1:]
                        inverse_ops.append(('resize', spatial_shape.copy()))
                        padded = [((d + k - 1) // k) * k for d in spatial_shape]
                        current_shape_with_channel = [current_shape_with_channel[0]] + padded

                def apply_ops(tensor):
                    """Apply the inverse op chain (in reverse order) to one tensor."""
                    for op_type, op_param in reversed(inverse_ops):
                        if op_type == 'resize':
                            target_shape = list(op_param)
                            resize_transform = ResizeWithPadOrCrop(spatial_size=target_shape)
                            resized = [resize_transform(tensor[i]) for i in range(tensor.shape[0])]
                            tensor = torch.stack(resized)

                        elif op_type == 'flip':
                            flip_dim = 2 + op_param
                            tensor = torch.flip(tensor, dims=[flip_dim])

                        elif op_type == 'transpose':
                            # Undo transpose - apply the inverse permutation
                            indices = op_param
                            inverse_indices = [0] * len(indices)
                            for i, idx in enumerate(indices):
                                inverse_indices[idx] = i

                            # For [0,2,1], inverse is also [0,2,1] (self-inverse)
                            if len(indices) == 3 and indices == [0, 2, 1]:
                                tensor = tensor.transpose(2, 3)
                            else:
                                spatial_inverse = inverse_indices[1:]  # Skip channel dim
                                perm = [0, 1] + [i + 2 for i in spatial_inverse]
                                tensor = tensor.permute(perm)
                    return tensor

                # Apply to output, and to target so metrics are computed at original size
                batch.output = apply_ops(batch.output)
                if hasattr(batch, 'target') and batch.target is not None:
                    batch.target = apply_ops(batch.target)

        except Exception as e:
            import traceback
            print(f"Warning: Could not apply inverse transforms: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            print("Continuing without inverse transforms...")

        return batch

    def restore_from_metadata(self, output, original_spatial_shape):
        """Restore output to original size using metadata"""
        # Handle tensor format - original_spatial_shape is tensor([[150, 233, 205]])
        if isinstance(original_spatial_shape, torch.Tensor):
            if original_spatial_shape.dim() == 2:  # Shape is [[150, 233, 205]]
                original_shape = original_spatial_shape[0].cpu().numpy()
            else:  # Shape is [150, 233, 205]
                original_shape = original_spatial_shape.cpu().numpy()
        else:
            original_shape = original_spatial_shape

        current_shape = output.shape[2:]  # Skip batch and channel dimensions

        print(f"Restoring from {current_shape} to {original_shape}")

        # Calculate cropping slices
        slices = [slice(None), slice(None)]  # Keep batch and channel dimensions

        for orig_size, curr_size in zip(original_shape, current_shape):
            if curr_size > orig_size:
                # Calculate symmetric crop (assuming symmetric padding was used)
                total_padding = curr_size - orig_size
                crop_start = total_padding // 2
                crop_end = crop_start + orig_size
                slices.append(slice(crop_start, crop_end))
                print(f"Cropping dimension: {curr_size} -> {orig_size} (crop {crop_start}:{crop_end})")
            else:
                slices.append(slice(None))

        restored = output[tuple(slices)]
        print(f"Successfully restored shape: {output.shape} -> {restored.shape}")
        return restored

    def test(self):
        while self.data.eval is not None:
            self.eval()
            if self.args.jsonData is not None:
                break
            self.data.reload_new_data()
        # if self.args.hasTarget:
        #     self.write_quantitative()

    def initialize_outputs(self):
        keys = self.args.mod
        case_outputs = {}
        case_sizes = self.get_case_sizes_from_json()

        # Track our position in the dataset
        current_idx = 0

        # Initialize case outputs using the first slice of each case block
        for case_id, num_slices in case_sizes.items():
            try:
                # Get sample for this case using the current_idx
                sample_data = self.data.eval.dataset[current_idx]
                case_sample = sample_data['input']
                empty_mat = []
                input_dims = len(case_sample.shape)

                # Spatial dims of the transformed (padded/resized) sample - fallback
                transformed_dims = [case_sample.size(d) for d in range(1, input_dims)]

                # Get dimensions - use original if inverse transforms are enabled
                dims = transformed_dims
                if hasattr(self.args, 'apply_inverse_transforms') and self.args.apply_inverse_transforms:
                    from monai.data import MetaTensor
                    # Get original spatial shape from metadata
                    if isinstance(case_sample, MetaTensor) and hasattr(case_sample, 'meta'):
                        original_shape = case_sample.meta.get('spatial_shape', None)
                        if isinstance(original_shape, torch.Tensor):
                            if original_shape.dim() == 2:
                                original_shape = original_shape[0].cpu().tolist()
                            else:
                                original_shape = original_shape.cpu().tolist()

                        if original_shape is not None:
                            # Inverse transforms restore the original dimensions
                            dims = list(original_shape)
                            print(f"Using original dimensions for case {case_id}: {dims}")

                for key in keys:
                    if input_dims == 3:  # 2D case (B,H,W)
                        mat = torch.empty(num_slices, dims[0], dims[1])
                    elif input_dims == 4:  # 3D case (B,H,W,D)
                        mat = torch.empty(num_slices, dims[0], dims[1], dims[2])
                    empty_mat.append(mat)

                case_outputs[case_id] = dict(zip(keys, empty_mat))

                # Move index to start of next case block
                current_idx += num_slices

            except Exception as e:
                import traceback
                print(f"Error initializing case {case_id}: {e}")
                print(f"Full traceback:\n{traceback.format_exc()}")
                continue

        return case_outputs

    def get_case_sizes_from_json(self):
        """Count number of slices per case from JSON data."""
        evalkey = self.args.keyTest if self.args.test else 'validation'
        if self.args.jsonData.endswith(".pkl"):
            with open(self.args.jsonData, 'rb') as f:
                data = pickle.load(f)[evalkey]
        else:
            with open(self.args.jsonData, 'r') as f:
                data = json.load(f)[evalkey]

        case_sizes = defaultdict(int)
        current_case = None
        current_count = 0

        # Assuming data is ordered by case
        for item in data:
            path = item['input']
            case_num = self.extract_case_number(path)

            if current_case != case_num:
                if current_case is not None:
                    case_sizes[current_case] = current_count
                current_case = case_num
                current_count = 1
            else:
                current_count += 1

        # Don't forget to add the last case
        if current_case is not None:
            case_sizes[current_case] = current_count

        return dict(case_sizes)


    def extract_case_number(self, path):
        """Extract case number based on path structure."""
        import re
        parts = path.split('/')

        # Standard 2D case: numeric folder 3rd from end (e.g., /data/42/slice.png)
        if len(parts) >= 3:
            try:
                return int(parts[-3])
            except ValueError:
                pass

        # Volumetric case: extract numbers from folder name (e.g., BraTS-GLI-01337-000)
        folder = parts[-2] if len(parts) >= 2 else parts[-1]
        nums = re.findall(r'\d+', folder)
        if len(nums) >= 2:
            return int(nums[-2] + nums[-1])
        elif nums:
            return int(nums[-1])
        return abs(hash(folder)) % 1000000

    def process_output_train(self):
        if self.args.metricForSaving is None:
            return
        metrics = self.metricStats['Train']
        if isinstance(metrics[self.args.epoch][self.args.metricForSaving][0],
                      torch.Tensor):
            Stats = [x.mean() for x in
                     metrics[self.args.epoch][self.args.metricForSaving]]
        else:
            Stats = metrics[self.args.epoch][self.args.metricForSaving]
        metricForSavingStat = np.mean(Stats)
        if self.bestEpochMetric < metricForSavingStat:
            self.bestEpochMetric = metricForSavingStat
            print(f'New best {self.args.metricForSaving} = '
                  f'{self.bestEpochMetric} for epoch {self.args.epoch}')
            self.save_model()

    def process_output_eval(self):
        if not self.args.test:
            self.save_best_epoch()

    def save_best_epoch(self):

        if self.args.metricForSaving is None:
            return
        metrics = self.metricStats['Validation'] if self.args.val else self.metricStats['Test']  # change to
        if isinstance(metrics[self.args.epoch][self.args.metricForSaving][0],
                      torch.Tensor):
            Stats = [x.mean() for x in
                     metrics[self.args.epoch][self.args.metricForSaving]]
        else:
            Stats = metrics[self.args.epoch][self.args.metricForSaving]
        metricForSavingStat = np.mean(Stats)
        if self.bestEpochMetric < metricForSavingStat:
            self.bestEpochMetric = metricForSavingStat
            print(f'New best {self.args.metricForSaving} = '
                  f'{self.bestEpochMetric} for epoch {self.args.epoch}')
            self.save_model()


    def save_model(self):
        print('saving model to file')
        torch.save(self.model.state_dict(), self.modelPath)
        print(f'model saved in {self.modelPath.as_posix()}')

    def write_quantitative(self):

        self.excelRowNames = list(np.unique(self.excelRowNames))
        if len(self.excelRowNames):
            df = pd.DataFrame(self.excelCols, index=self.excelRowNames)
        else:
            df = pd.DataFrame(self.excelCols)
        df.to_excel(self.excelPath.as_posix())
        print(f'excel saved to {self.excelPath}')

    def process_output_batch_eval(self, batch, catOutputs=None):
        if self.args.test:
            catOutputs = self.process_output_batch_test(batch, catOutputs)
        if type(batch.output) is list:
            if self.args.cnnModule == 'FSMNet':
                batch.output = batch.output[0:1]
            batch.output = self.convert_list_to_tensor(batch.output)
        self.calc_metrics(batch, training=False)
        return catOutputs

    def load_checkpoint(self):
        if self.args.test:
            if self.args.loadCheckpointInTest:
                ckpt = torch.load(self.ckptPath.as_posix(),weights_only=False)
                modelDict = ckpt['state_dict']
                loadedPath = f'{self.ckptPath} (epoch {ckpt["epoch"]})'
            else:
                modelDict = torch.load(self.modelPath.as_posix(),weights_only=False)
                loadedPath = self.modelPath
            modelDict = self.fix_dict_if_needed(modelDict)
            self.model.load_state_dict(modelDict)
            print(f'model loaded from: {loadedPath}')
        elif self.ckptPath.exists() and self.args.loadCheckpoint:
            ckpt = torch.load(self.ckptPath.as_posix(),weights_only=False)

            # First load model state
            self.model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            # Don't try to load optimizer state from previous training
            # Just use the new optimizer configuration
            self.args.epoch = ckpt['epoch'] + 1
            self.bestEpochMetric = ckpt.get('bestMetric', 0)
            if self.args.cosineAnneal:
                if 'scheduler' in ckpt:
                    self.scheduler.load_state_dict(ckpt['scheduler'])
                else:
                    print('No scheduler state found in checkpoint, initializing cosine annealing from scratch')
            # Optionally override the base LR on resume: the checkpoint restores the
            # old optimizer/scheduler LR, so bumping learningRate in the config has no
            # effect unless we explicitly reset the base LR here.
            if self.args.get('overrideLrOnResume', False):
                new_lr = self.args.learningRate
                for pg in self.optimizer.param_groups:
                    pg['lr'] = new_lr
                    pg['initial_lr'] = new_lr
                if self.args.cosineAnneal and hasattr(self.scheduler, 'base_lrs'):
                    self.scheduler.base_lrs = [new_lr for _ in self.optimizer.param_groups]
                print(f'overriding base LR on resume -> {new_lr}')
            print(f'checkpoint loaded, epoch = {self.args.epoch}')



    def fix_dict_if_needed(self, Dict):

        newDict = OrderedDict()
        for k in Dict:
            if k.split('.')[0] != 'model':
                newDict['model.' + k] = Dict[k]

        if len(newDict):
            return newDict
        else:
            return Dict

    def process_output_batch_test(self, batch, catOutputs):
        """Process batch outputs for testing, filling sequentially."""
        # Find which case we should be filling based on self.t
        current_case = None
        accumulated_count = 0

        # Iterate through cases in order to find where this slice belongs
        for case_id, case_outputs in catOutputs.items():
            case_size = case_outputs[list(case_outputs.keys())[0]].size(0)  # Get size from first modality
            if self.t < accumulated_count + case_size:
                current_case = case_id
                # Adjust t to be relative to current case
                self.case_t = self.t - accumulated_count
                break
            accumulated_count += case_size

        if current_case is None:
            raise ValueError(f"Index {self.t} exceeds total number of slices")

        # Process the output for the current case
        for index, key in enumerate(catOutputs[current_case].keys()):
            if isinstance(batch.output, list):
                output = batch.output[index][0, 0]  # First sample, first channel
            else:
                output = batch.output[0, index]  # First sample, indexed channel

            if output.dim() == 2:
                catOutputs[current_case][key][self.case_t, :, :] = output
            elif output.dim() == 3:
                catOutputs[current_case][key][self.case_t, :, :, :] = output

        return catOutputs
    # def process_output_batch_test(self, batch, catOutputs):
    #     for index, key in enumerate(catOutputs.keys()):
    #         if isinstance(batch.output, list):
    #             catOutputs[key][self.t, :, :] = batch.output[index][0, 0, :, :]
    #         else:
    #             catOutputs[key][self.t, :, :] = batch.output[0, index, :, :]
    #     #catOutputs['mod2'][self.t, :, :] = batch.output[1][0, 0, :, :]
    #
    #     return catOutputs

    def nifti_results(self, catOutputs):
        # Load the pickle file
        # with open(self.args.jsonData, 'rb') as f:
        #     load = pickle.load(f)
        # # Loop over the cases in the pickle file
        # # Loop over the cases in the pickle file
        # #case_duplication = '-1'
        # for i in range(catOutputs[next(iter(catOutputs.keys()))].shape[0]):
        #     # Extract the case path from the pickle data
        #
        #     case_path = load['test'][i]['input']
        #     case_number = os.path.dirname(os.path.dirname(case_path))
        result_dir = os.path.join(Path(__file__).parent, 'results', self.args.project_name)
        os.makedirs(result_dir, exist_ok=True)
        result_folder = os.path.join(result_dir, self.args.conf + '.npy')
        np.save(result_folder, catOutputs)

            # Loop over each modality in catOutputs
            # for key in catOutputs.keys():
            #     # Read the original NIfTI file to get the affine matrix
            #     output_file_path = os.path.join(case_number, key + ' deblur')
            #     if not os.path.exists(output_file_path):
            #         os.makedirs(output_file_path)
            #     npy_file_path = os.path.join(output_file_path, key +'_deblurred.npy')
            #     np.save(npy_file_path, catOutputs[key])

    def dicom_results(self, catOutputs):
        if self.args.test_save:
            # Determine if we're working with BRATS (NPY) or not (DICOM)
            is_brats = hasattr(self.args, 'project_name') and ( 'BRATS' in self.args.project_name or self.args.use_npy)

            # Initialize DICOM handler if needed
            if not is_brats:
                dcm = myDicom()

            # Iterate through each case
            for case_id in catOutputs.keys():
                case_outputs = catOutputs[case_id]

                # Iterate through each modality
                for index, key in enumerate(case_outputs.keys()):
                    data = case_outputs[key]
                    if data.dim() == 4:  # If 4D, remove batch dimension
                        data = data.squeeze(0)

                    # Use the existing output path structure
                    # Get case number from the path (number after the first '/' from the end)
                    path_parts = self.out_path[index].rstrip('/').split('/')
                    path_parts[-2] = str(case_id)  # Replace the case number
                    out_path = '/'.join(path_parts)

                    if is_brats:
                        # NPY handling for BRATS project
                        # Create the output directory if it doesn't exist
                        os.makedirs(out_path, exist_ok=True)

                        # Save as .npy file
                        npy_file_path = os.path.join(out_path, f"{key}.npy")
                        np.save(npy_file_path, data.numpy())

                        print(f"Saved NPY file: {npy_file_path}")
                    else:
                        # DICOM handling for other projects
                        # Do the same for reference path
                        ref_parts = self.ref_path[index].rstrip('/').split('/')
                        ref_parts[-2] = str(case_id)
                        ref_path = '/'.join(ref_parts)

                        dcm.writeDicomSeries(
                            output_path=out_path,
                            source_path=ref_path,
                            I=data.numpy(),
                            description=f"{self.args.description} {self.args.mod[index]}"
                        )

                        print(f"Saved DICOM series to: {out_path}")

        del catOutputs


    def test_dirs(self):
        a = json.load(open(self.args.jsonData, 'r'))
        path_out = a['test'][0]['input']
        # 2D slice entries: 'case/subdir/file.ext' → two dirname calls → 'case'
        # 3D volume entries: 'case/subdir' (no extension) → one dirname call → 'case'
        if os.path.splitext(os.path.basename(path_out))[1]:
            path_out = os.path.dirname(os.path.dirname(path_out))
        else:
            path_out = os.path.dirname(path_out)
        project_root = Path(__file__).parent
        out_project = getattr(self.args, 'test_project', None) or self.args.project_name
        self.out_path = [os.path.join(project_root, 'results', out_project, self.args.conf, os.path.basename(path_out),
                                      self.args.mod[i]) + '_Deblur' for i in range(len(self.args.mod))]
        # ref_path is the SOURCE series dir (used to copy DICOM headers when writing
        # DICOM output). The json stores paths relative to base_dir, so resolve it.
        from datasets.monai_dataset import _resolve_base_dir
        base = _resolve_base_dir(self.args.jsonData, split='test')
        ref_root = os.path.join(str(base), path_out) if base is not None else path_out
        self.ref_path = [os.path.join(ref_root,
                                      self.args.mod[i]) for i in range(len(self.args.mod))]


    def modality_log_metric(self, batch, training=False):
        x = self.pre_metric_calculation(batch, training)
        if x is not None and \
                (('confMat' in x and x['confMat'].sum() > 0)
                 or 'confMat' not in x):
            for key in x:
                for metric in self.args.metrics:
                    if 'monai' in metric:
                        real_metric = metric.replace('monai_', '')
                        _m = Metrics.monai.metrics.compute_confusion_matrix_metric(confusion_matrix=x[key]['confMat'],
                                                                                   metric_name=real_metric)
                        if _m[0].isnan().sum() == 0:
                            _m = _m[x[key]['batch'].target[:, 0]]
                        _m = _m[~torch.isnan(_m)]
                        _m = _m.to('cpu').detach()
                    else:
                        _m = self.metricFuncs[metric](x[key]).to('cpu').detach()
                    _m = self.post_metric_calculation(_m, **x[key])
                    metric_name = metric \
                        if 'monai' not in metric \
                        else real_metric
                    self.metricStats[key][metric_name][self.t].append(_m)

    def save_results_dict(self):
        # Serialize data into file:
        torch.save(self.metricStats, self.resultLogPath, pickle_module=dill)

    def calc_mean(self):
        # Initialize 'metrics' with a nested defaultdict structure
        metrics = defaultdict(lambda: defaultdict(lambda: {'mean': None, 'std': None}))

        for val in self.metricStats:
            for metric, values in self.metricStats[val].items():
                # Convert MetaTensor to numpy and filter out nan values
                filtered_values = [value[0].numpy() for value in values.values() if
                                   not (np.isnan(value[0].numpy()) or np.isinf(value[0].numpy()))]

                if filtered_values:  # Ensure there are values before calculating mean and std
                    metrics[val][metric]['mean'] = sum(filtered_values) / len(filtered_values)
                    metrics[val][metric]['std'] = np.std(filtered_values, ddof=0)  # Use ddof=0 for population std
                else:
                    # If all values are nan, mean and std remain as None
                    continue  # Or explicitly set to None, but it's already the default
        wandb.log(metrics)
        return metrics
