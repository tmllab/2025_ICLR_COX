import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models import lora as LoRA
from models.load_models import load_imagebind
from models.vib import VIB
from models.utils import save_module
from utils import cal_metric_and_print, Vector
import lightning as L


find_MM_models = {
    'imagebind': load_imagebind,
}

def agreement(logits1, logits2, topk=8):
    logits1, logits2 = logits1.detach(), logits2.detach()
    soft_1 = nn.functional.softmax(logits1, dim=1)
    soft_2 = nn.functional.softmax(logits2, dim=1)
    entropy_1 = - soft_1 * nn.functional.log_softmax(logits1, dim=1)
    entropy_2 = - soft_2 * nn.functional.log_softmax(logits2, dim=1)
    entropy_1 = torch.sum(entropy_1, dim=1)
    entropy_2 = torch.sum(entropy_2, dim=1)

    discrepancy = torch.nn.ReLU()(-torch.mean(entropy_1 - entropy_2))
    indices = torch.topk(discrepancy, topk)
    logits = (logits1 + logits2) / 2.
    label = torch.argmax(logits, dim=1).item()

    return label, indices

def disagreement(logits1, logits2, m=1.0):
    logits1, logits2 = logits1.detach(), logits2.detach()
    soft_1 = nn.functional.softmax(logits1, dim=1)
    soft_2 = nn.functional.softmax(logits2, dim=1)
    entropy_1 = - soft_1 * nn.functional.log_softmax(logits1, dim=1)
    entropy_2 = - soft_2 * nn.functional.log_softmax(logits2, dim=1)
    entropy_1 = torch.sum(entropy_1, dim=1)
    entropy_2 = torch.sum(entropy_2, dim=1)

    discrepancy = torch.nn.ReLU()(m - torch.mean(entropy_1 - entropy_2))
    indices = torch.topk(discrepancy, largest=False)

    return indices

class BindModel(L.LightningModule):
    def __init__(
        self,
        args,
        model_type,
        modalities,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.modalities = modalities
        self.im_modalities = modalities.remove(args.oom)
        self.oom = args.oom
        self.models = find_MM_models[model_type](args, modalities)
        self.label_distribution = Vector(args.num_classes)

        self.initialize_oom_learner(args.oom)
        if args.lora:
            self.models.modality_preprocessors[args.oom].requires_grad_(False)
            self.models.modality_trunks[args.oom].requires_grad_(False)

            self.models.modality_trunks[args.oom].update(LoRA.apply_lora_modality_trunks(self.models.modality_trunks[args.oom],
                rank=args.lora_rank, modality_names=modalities))

        elif args.linear_probing:
            self.models.modality_preprocessors[args.oom].requires_grad_(False)
            self.models.modality_trunks[args.oom].requires_grad_(False)
            self.models.modality_postprocessors[args.oom].requires_grad_(False)

            self.models.modality_heads[args.oom].requires_grad_(False)
            final_layer = list(self.models.modality_heads[args.oom])[-1]
            final_layer.requires_grad_(True)

        classifiers = {
            self.im_modalities[0]: nn.Linear(args.out_embed_dim, args.num_classes),
            self.im_modalities[1]: nn.Linear(args.out_embed_dim, args.num_classes),
            self.oom: nn.Linear(args.out_embed_dim, args.num_classes)
        }
        vibs = {
            self.im_modalities[0]: VIB(1024, 256),
            self.im_modalities[1]: VIB(1024, 256),
        }
        self.classifiers = nn.ModuleDict(classifiers)
        self.vibs = nn.ModuleDict(vibs)

        # Total number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params}")

        # Number of trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

    def initialize_oom_learner(self, oom):
        self.models.modality_trunks[oom].apply(self._init_weights)
        self.models.modality_heads[oom].apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.args.lr, weight_decay=self.hparams.args.weight_decay, 
                                betas=self.hparams.args.momentum_betas)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.args.max_epoch, eta_min=self.hparams.args.lr / 50
        )
        return [optimizer], [lr_scheduler]


    def cox(self, sample, mode='train'):
        im_labels = sample.pop('im_label')
        oom_labels = sample.pop('oom_label')
        dataset_idx = sample.pop('dataset_idx')
        labeled_indices = oom_labels != -1
        outputs = self.models(sample)
        
        if mode == 'train':
            im_output1 = outputs[self.im_modalities[0]]
            im_output2 = outputs[self.im_modalities[1]]
            oom_output = outputs[self.oom]

            l_im_output1 = im_output1[labeled_indices]
            l_im_output2 = im_output2[labeled_indices]
            l_oom_output = oom_output[labeled_indices]
            l_im_map1 = self.vibs[self.im_modalities[0]](l_im_output1)
            l_im_map2 = self.vibs[self.im_modalities[1]](l_im_output2)

            oom_logits = self.classifiers[self.oom](oom_output)
            im_logits1 = self.classifiers[self.im_modalities[0]](im_output1)
            im_logits2 = self.classifiers[self.im_modalities[1]](im_output2)
            
            recons_loss = F.mse_loss(l_im_map1, l_oom_output)
            recons_loss += F.mse_loss(l_im_map2, l_oom_output)
            class_loss = F.cross_entropy(oom_logits, oom_labels, ignore_index=-1)
            class_loss += 0.5 * F.cross_entropy(im_logits1, im_labels)
            class_loss += 0.5 * F.cross_entropy(im_logits2, im_labels)

            if self.current_epoch > 5:
                im_pred1 = self.classifiers[self.im_modalities[0]](oom_output)
                im_pred2 = self.classifiers[self.im_modalities[1]](oom_output)
                agree_label, agree_indices = agreement(im_pred1, im_pred2)
                agree_logits = self.classifiers[self.oom](oom_output[agree_indices])
                self.label_distribution.update(agree_logits)
                class_loss += F.cross_entropy(agree_logits, agree_label)

                disagree_indices = disagreement(im_pred1, im_pred2)
                disagree_logits_mean = self.classifiers[self.oom](oom_output[disagree_indices]).mean(0)
                class_loss += F.kl_div(disagree_logits_mean, self.label_distribution.compute())

            loss = class_loss + recons_loss
            return loss
        else:
            return outputs


    def training_step(self, batch):
        loss = self.cox(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.cox(batch, mode="val")
        for key, value in outputs.items():
            self.val_embeddings[key].append(value)

    def on_validation_epoch_start(self):
        self.val_embeddings = {mod: [] for mod in self.modalities}

    def on_validation_epoch_end(self):
        epoch_embeddings = {}
        for key, values in self.val_embeddings.items():
            epoch_embeddings[key] = torch.cat(values, dim=0)
        metrics = cal_metric_and_print(epoch_embeddings, print_metric=True)

        for key, value in metrics.items():
            for criteria, result in value.items():
                self.log("test_" + key + "_" + criteria, result, prog_bar=True,
                    on_step=False, on_epoch=True)

        if self.hparams.args.save_models:
            if self.hparams.args.lora:
                # Save LoRA checkpoint
                LoRA.save_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=self.hparams.args.checkpoints, 
                            postfix='test')
                # Save postprocessors & heads
                save_module(self.model.modality_postprocessors, module_name="postprocessors",
                            checkpoint_dir=self.hparams.args.checkpoints,
                            postfix='test')
                save_module(self.model.modality_heads, module_name="heads",
                            checkpoint_dir=self.hparams.args.checkpoints,
                            postfix='test')
            elif self.hparams.args.linear_probing:
                # Save postprocessors & heads
                save_module(self.model.modality_heads, module_name="heads",
                            checkpoint_dir=self.hparams.args.checkpoints,
                            postfix='test')

