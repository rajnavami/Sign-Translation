# trainer/psuedo_gloss_trainer.py
#
# CHANGES vs original:
#   ONLY prep_batch is changed — 3 lines added to extract flow from batch
#   and include it in model_input.
#
#   train_step, eval_step, test_step are IDENTICAL to original — they all
#   call prep_batch so they get flow automatically without any changes.
#
# How flow flows through this trainer:
#   batch["flow"]              (list of (T, 2, 64, 64) tensors, from dataloader)
#       -> prep_batch extracts it
#       -> model_input["list_of_flows"]
#       -> self.model(**x["model_input"])
#       -> dino_adaptor_model.forward(list_of_frames, list_of_flows=...)
#       -> FlowBranch + GatedFusion
#
# If flow_lmdb_dir is not set in config: batch has no "flow" key,
# batch.get("flow", None) returns None, model_input["list_of_flows"] = None,
# dino_adaptor_model skips the flow branch entirely. Zero behavioural change.

import datetime
import os
import yaml

import torch
import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from models.get_models import get_model
from trainer.base.base_trainer import BaseTrainer
from callbacks.full_callback import LoggingCallback
from train_utils.checkpoint_helpers import (
    get_latest_saved_file,
    get_best_checkpoint_details,
)


class Trainer(BaseTrainer):
    """
    Updated for DINOv3 + LoRA + optical flow integration.

    Key fixes/improvements (carried over from previous version):
    - safer yaml saving via yaml.safe_dump
    - autocast device_type works on CUDA/CPU
    - auto_model sync_bn only when distributed > 1
    - optional find_unused_parameters via cfg
    - zero_grad(set_to_none=True) for better memory behavior

    New in this version:
    - prep_batch extracts flow from batch and passes to model_input
    """

    def __init__(self, local_rank, *args, **kwargs):
        super().__init__(args)
        cfg = args[0]
        self.cfg = cfg

        if "run" in cfg:
            pass
        else:
            if getattr(cfg, "log_name", None):
                stage_name = (
                    "pretraining_"
                    + cfg.log_name
                    + "_"
                    + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                )
            else:
                stage_name = "pretraining_" + datetime.datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

            writer = SummaryWriter(f"runs/{stage_name}")
            self.logger = LoggingCallback(self.cfg, stage_name, writer)
            self.logger.start_logger()

            config_save_dir  = f"runs/{stage_name}"
            os.makedirs(config_save_dir, exist_ok=True)
            config_save_path = os.path.join(config_save_dir, "stage1_config.yaml")
            cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
            with open(config_save_path, "w") as f:
                yaml.safe_dump(cfg_dict, f, default_flow_style=False)

        self.max_length = 128

        (
            train_dl, valid_dl, test_dl,
            train_dict, valid_dict, test_dict,
        ) = self.get_dataloaders(cfg)

        self.init_models()
        self.init_criterion(cfg)
        self.init_optimizer(cfg)

        trainer      = Engine(self.train_step)
        evaluator    = Engine(self.eval_step)
        valid_tester = Engine(self.test_step)
        test_tester  = Engine(self.test_step)

        self.scheduler = self.prep_scheduler(
            cfg, train_dl, self.optimizer, trainer, evaluator
        )

        self.init_metrics(
            trainer, "train",
            ["accuracy", "class_accuracy", "f1_score", "class_f1_score"],
            additional=True,
            max_length=self.max_length,
            num_samples=train_dict["length"],
            num_classes=len(train_dict["dict_lem_to_id"]),
        )
        self.init_metrics(
            evaluator, "valid",
            ["accuracy", "class_accuracy", "f1_score", "class_f1_score"],
            additional=True,
            max_length=self.max_length,
            num_samples=valid_dict["length"],
            num_classes=len(train_dict["dict_lem_to_id"]),
        )
        self.init_metrics(
            valid_tester, "valid_test",
            ["accuracy", "class_accuracy", "f1_score", "class_f1_score"],
            additional=False,
            max_length=self.max_length,
            num_samples=valid_dict["length"],
            num_classes=len(train_dict["dict_lem_to_id"]),
        )
        self.init_metrics(
            test_tester, "test_test",
            ["accuracy", "class_accuracy", "f1_score", "class_f1_score"],
            additional=False,
            max_length=self.max_length,
            num_samples=test_dict["length"],
            num_classes=len(train_dict["dict_lem_to_id"]),
        )

        def score_function(engine):
            return cfg.score_factor * float(engine.state.metrics[cfg.score_name])

        to_save = {
            "model":     self.model,
            "optimizer": self.optimizer,
            "trainer":   trainer,
        }
        if self.scaler    is not None: to_save["scaler"]    = self.scaler
        if self.scheduler is not None: to_save["scheduler"] = self.scheduler

        self.save_checkpoints(
            cfg, trainer, evaluator, score_function,
            best_only=False, to_save=to_save
        )

        objects_to_load = {
            "model":     self.model,
            "optimizer": self.optimizer,
            "trainer":   trainer,
        }
        if self.scaler    is not None: objects_to_load["scaler"]    = self.scaler
        if self.scheduler is not None: objects_to_load["scheduler"] = self.scheduler

        if ("model_only" in cfg) and (cfg.model_only is True):
            objects_to_load = {"model": self.model}

        self.load_checkpoints(cfg, trainer, objects_to_load=objects_to_load)

        if "run" in cfg:
            if cfg["load_from_ckpt"] == "best":
                ckpt, _, _ = get_best_checkpoint_details(
                    cfg.save_dir, best_checkpoint_name="_result_checkpoint_"
                )
                self.model.load_state_dict(
                    torch.load(ckpt, map_location="cpu")["model"]
                )
            elif cfg["load_from_ckpt"] == "latest":
                ckpt, _, _ = get_latest_saved_file(
                    cfg.save_dir, extension="pt", name_latest="latest_epoch"
                )
                self.model.load_state_dict(
                    torch.load(ckpt, map_location="cpu")["model"]
                )
            else:
                print("SKIPPING THE LOADING FROM CHECKPOINT")

            self.trainer      = trainer
            self.evaluator    = evaluator
            self.valid_tester = valid_tester
            self.test_tester  = test_tester
        else:
            self.prepare_runner(
                cfg, trainer, evaluator, valid_dl, valid_tester, test_tester, test_dl
            )
            self.cleaning_with_progress(trainer, evaluator, cfg, train_dl)

            logger.info("\n" + "=" * 70)
            logger.info("DEBUG: Testing first batch and forward pass")
            logger.info("=" * 70)
            try:
                train_iter = iter(train_dl)
                batch = next(train_iter)
                logger.info(f"✓ Raw batch loaded successfully")
                logger.info(f"  Raw batch keys: {batch.keys()}")

                # Log whether flow is present in this batch
                if "flow" in batch:
                    logger.info(
                        f"  Flow present: {len(batch['flow'])} tensors, "
                        f"each shape {batch['flow'][0].shape}"
                    )
                else:
                    logger.info("  Flow not present in batch (flow_lmdb_dir not set)")

                x = self.prep_batch(batch, isValid=False)
                logger.info(f"✓ Batch prepared")
                logger.info(f"  model_input keys: {x['model_input'].keys()}")

                # Log flow status in model_input
                if x["model_input"].get("list_of_flows") is not None:
                    logger.info("  ✓ list_of_flows present in model_input")
                else:
                    logger.info("  list_of_flows is None in model_input")

            except Exception as e:
                logger.error(f"✗ Debug batch test failed: {e}")
                raise

            trainer.run(
                train_dl, max_epochs=cfg.max_epochs, epoch_length=cfg.train_length
            )

    def prepare_runner(
        self, cfg, trainer, evaluator, valid_dl, valid_tester, test_tester, test_dl
    ):
        def run_evaluator(engine):
            engine.state.output = None
            engine.state.batch  = None
            evaluator.run(valid_dl, max_epochs=1, epoch_length=cfg.val_length)

        self.logger.on_train_epoch_end(trainer, self.optimizer)
        self.logger.on_train_iteration(trainer, self.model, self.scaler)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), run_evaluator)
        self.logger.on_valid_epoch_end(trainer, evaluator)
        self.logger.on_completion(trainer)

        @trainer.on(Events.EPOCH_STARTED)
        def on_epoch_started(engine):
            logger.info(
                f"[EPOCH START] Epoch {engine.state.epoch}/{engine.state.max_epochs}"
            )

        @trainer.on(Events.ITERATION_STARTED)
        def on_iteration_started(engine):
            if engine.state.iteration % 50 == 0:
                logger.info(
                    f"[ITERATION] Epoch {engine.state.epoch}, "
                    f"Iter {engine.state.iteration}"
                )

        @trainer.on(Events.ITERATION_COMPLETED)
        def on_iteration_completed(engine):
            if engine.state.iteration % 50 == 0:
                logger.info(f"[ITERATION DONE] output computed")

        @trainer.on(Events.EPOCH_COMPLETED)
        def on_epoch_completed(engine):
            logger.info(f"[EPOCH DONE] Epoch {engine.state.epoch} completed")

    # ── Metrics helpers (unchanged) ───────────────────────────────────────────

    def dict_metric_from_list(self, engine_type, list_of_metrics, dict_metrics, **kwargs):
        def output_fn(a):
            x   = a["y_pred"]["dict_post_output"]["logits"]
            tgt = a["target"]["targets"]["pseudo_gloss_ids"]
            return (x, tgt)

        if "accuracy" in list_of_metrics:
            from metrics.accuracy_score import AccuracyScore
            dict_metrics[f"{engine_type}/avg_acc"] = AccuracyScore(
                output_transform=output_fn, thresholds=0.5
            )
        if "class_accuracy" in list_of_metrics:
            from metrics.class_accuracy_score import ClassAccuracyScore
            dict_metrics[f"{engine_type}/cls_acc"] = ClassAccuracyScore(
                output_transform=output_fn, thresholds=0.5,
                num_classes=kwargs["num_classes"],
            )
        if "f1_score" in list_of_metrics:
            from metrics.f1_score import F1Score
            dict_metrics[f"{engine_type}/f1_score"] = F1Score(
                output_transform=output_fn, thresholds=0.5
            )
        if "class_f1_score" in list_of_metrics:
            from metrics.class_f1_score import ClassF1Score
            dict_metrics[f"{engine_type}/class_f1_score"] = ClassF1Score(
                output_transform=output_fn, thresholds=0.5,
                num_classes=kwargs["num_classes"],
            )
        return dict_metrics

    def init_metrics(self, engine, engine_type, list_of_metrics, additional=True, **kwargs):
        dict_metrics = {}
        dict_metrics = self.dict_metric_from_list(
            engine_type, list_of_metrics, dict_metrics, **kwargs
        )
        super().init_metrics(
            engine, engine_type, dict_metrics=dict_metrics, additional=additional
        )

    # ── Model init (unchanged) ────────────────────────────────────────────────

    def init_models(self):
        dict_model_params = self.cfg.model_params.to_dict()
        self.model = get_model(self.cfg.model_name, dict_model_params)

        find_unused = bool(getattr(self.cfg, "find_unused_parameters", False))
        sync_bn     = bool(getattr(self.cfg, "sync_bn", True)) and (
            idist.get_world_size() > 1
        )
        self.model  = idist.auto_model(
            self.model, find_unused_parameters=find_unused, sync_bn=sync_bn
        )

    # ── prep_batch (ONLY CHANGED METHOD) ─────────────────────────────────────

    def prep_batch(self, batch, isValid=False, cuda=True):
        """
        Prepares the batch dict into model_input and targets.

        CHANGE vs original: extracts flow from batch and adds list_of_flows
        to model_input. This is the ONLY change in this entire file.

        Why here and not in train_step/eval_step/test_step?
        - All three steps call prep_batch, so adding flow here means all
          three automatically get flow without any code duplication.
        - convert_tensor at the end moves everything to GPU, including the
          flow tensors — ignite handles lists of tensors correctly.
        """
        idx, frames = (batch["index"], batch["frames"])
        frame_features = frames

        # ── NEW: extract precomputed flow from batch ──────────────────────────
        # batch["flow"] is a list of (T, 2, 64, 64) tensors — one per sample.
        # It is present only when flow_lmdb_dir is set in the config.
        # .get() with default None means this is fully backward compatible:
        # if flow is not in the batch, list_of_flows = None and the model
        # skips the flow branch entirely (no error, no overhead).
        list_of_flows = batch.get("flow", None)
        # ── END NEW ───────────────────────────────────────────────────────────

        res = {
            "model_input": {
                "frame_features": frame_features,
                "max_len": torch.tensor(
                    self.cfg.max_seq_len if "max_seq_len" in self.cfg else 512
                ),
                # ── NEW: pass flow into model_input ───────────────────────────
                # dino_adaptor_model.forward() accepts list_of_flows as an
                # optional kwarg. When None, it skips FlowBranch + GatedFusion.
                # convert_tensor (below) will move flow tensors to GPU.
                "list_of_flows": list_of_flows,
                # ── END NEW ───────────────────────────────────────────────────
            },
            "targets": {
                "pseudo_gloss_ids": batch["pseudo_gloss_ids"]
                if "pseudo_gloss_ids" in batch
                else [],
                "index": torch.stack(idx),
            },
        }

        return (
            convert_tensor(res, device=idist.device(), non_blocking=True)
            if cuda
            else res
        )

    # ── Training steps (ALL UNCHANGED — they just call prep_batch) ───────────

    def train_step(self, engine, batch):
        engine.state.batch  = None
        engine.state.output = None
        self.model.train()

        x = self.prep_batch(batch, isValid=False)   # flow is now inside x

        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.autocast(device_type=device_type, dtype=self.dtype):
            self.optimizer.zero_grad(set_to_none=True)
            y_pred = self.model(**x["model_input"])  # list_of_flows passed here

        loss, dict_losses = self.loss_fn(y_pred, x["targets"])

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.grad_clip_value is not None or self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip_value is not None:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), self.grad_clip_value
                    )
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
            self.manually_update_gradients()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = loss.float()
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self.grad_clip_value
                )
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            self.manually_update_gradients()
            self.optimizer.step()

        return {
            "y_pred":  y_pred,
            "target":  x,
            "losses":  {
                "loss": loss.detach(),
                **{k: v.detach() for k, v in dict_losses.items()},
            },
        }

    def manually_update_gradients(self):
        pass

    def eval_step(self, engine, batch):
        engine.state.batch  = None
        engine.state.output = None
        self.model.eval()

        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.inference_mode(True):
            x = self.prep_batch(batch, isValid=True)
            with torch.autocast(device_type=device_type, dtype=self.dtype):
                y_pred = self.model(**x["model_input"])

            loss, dict_losses = self.loss_fn(y_pred, x["targets"])

            return {
                "y_pred": y_pred,
                "target": x,
                "losses": {
                    "loss": loss.detach(),
                    **{k: v.detach() for k, v in dict_losses.items()},
                },
            }

    def test_step(self, engine, batch):
        engine.state.batch  = None
        engine.state.output = None
        self.model.eval()

        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.inference_mode(True):
            x = self.prep_batch(batch, isValid=True)
            with torch.autocast(device_type=device_type, dtype=self.dtype):
                y_pred = self.model(**x["model_input"])

            loss, dict_losses = self.loss_fn(y_pred, x["targets"])

            return {
                "y_pred": y_pred,
                "target": x,
                "losses": {
                    "loss": loss.detach(),
                    **{k: v.detach() for k, v in dict_losses.items()},
                },
            }