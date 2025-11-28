import os
import ignite.distributed as idist
import json
from ignite.engine import Engine, Events
import pprint
import collections
import time
import numpy as np
import torch
import io
import pickle
from pathlib import Path
import datetime


class LoggingCallback:

    def __init__(self, cfg, stage, writer):
        self.cfg = cfg
        Path("logs").mkdir(parents=True, exist_ok=True)

        self.tensor_writer = writer
        self.log_file_path = f"logs/training_{stage}_log.txt"

        if not os.path.exists(self.log_file_path) or os.path.getsize(self.log_file_path) == 0:
            with open(self.log_file_path, "a") as f:
                f.write(f"=== Training Log Started at {datetime.datetime.now()} ===\n\n")


    def start_logger(self):
        from environment_variables import CONFIG

        for key, value in CONFIG.items():
            os.environ[key] = value

        self.writer = None
        self.discord_hook = None
        if "discord" in self.cfg.logger_name and idist.get_rank() == 0:
            from train_utils.discord_hook import DiscordHook

            self.discord_hook = DiscordHook(
                title=self.cfg.name,
                project_name=self.cfg.project_name,
                discord_url=os.environ["DISCORD_URL"],
            )
        if "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
            import wandb

            self.writer = wandb.init(
                project=self.cfg.project_name,
                id=self.cfg.name,
                config=json.loads(self.cfg.to_json_best_effort()),
                save_code=False,
                tags=[],
                name=self.cfg.name,
                resume="allow",
                allow_val_change=True,
            )

    def on_train_epoch_end(self, trainer, optimizer):
        @trainer.on(Events.EPOCH_COMPLETED(every=1))
        def print_output(engine):
            if "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
                for k, v in engine.state.metrics.items():
                    if "summary" in k:
                        continue
                    if type(v) == dict:
                        for k_i, v_i in v.items():
                            self.writer.log(
                                {f"{k}_{k_i}": v_i, f"epoch": engine.state.epoch}
                            )
                    else:
                        self.writer.log({f"{k}": v, f"epoch": engine.state.epoch})
            if "discord" in self.cfg.logger_name and idist.get_rank() == 0:
                results = []
                for k, v in collections.OrderedDict(
                    sorted(engine.state.metrics.items())
                ).items():
                    if "summary" in k:
                        continue
                    if type(v) != dict:
                        results.append({"name": k, "value": v, "inline": True})
                self.discord_hook.send_message(
                    content=None,
                    description=f"Training Results for Epoch: {engine.state.epoch}",
                    results=results,
                    img=None,
                )

            if "text" in self.cfg.logger_name and idist.get_rank() == 0:
                print("TRAINER", engine.state.epoch)
                dict_res = {}
                for k, v in engine.state.metrics.items():
                    if "summary" in k:
                        continue
                    if type(v) == dict:
                        for k_i, v_i in v.items():
                            dict_res[f"{k}_{k_i}"] = v_i
                    else:
                        dict_res[f"{k}"] = v

                pprint.pprint(dict_res)

                ##
                with open(self.log_file_path, "a") as f:
                    f.write(f"\n[TRAIN EPOCH {engine.state.epoch}]\n")
                    pprint.pprint(dict_res, stream=f)
                    f.write("\n")
                ##
                if "tensorboard" in self.cfg.logger_name and idist.get_rank() == 0:
                    for key, value in dict_res.items():
                        self.tensor_writer.add_scalar(key, value, engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED(every=1))
        def print_lr(engine):
            for i, pg in enumerate(optimizer.param_groups):
                if "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
                    self.writer.log(
                        {
                            f"train/lr_{i}": optimizer.param_groups[i]["lr"],
                            f"epoch": engine.state.epoch,
                        }
                    )
                if "text" in self.cfg.logger_name and idist.get_rank() == 0:
                    lr_val = optimizer.param_groups[i]["lr"]
                    pprint.pprint({f"train/lr_{i}": lr_val})
                    ##
                    with open(self.log_file_path, "a") as f:
                        f.write(f"LR (group {i}): {lr_val}\n")
                    ##

    def on_train_iteration(self, trainer, model, scaler):
        @trainer.on(Events.ITERATION_COMPLETED(every=self.cfg.log_every))
        def print_iter_output(engine):
            if "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
                for k, v in engine.state.output["losses"].items():
                    if type(v) == dict:
                        for k_i, v_i in v.items():
                            self.writer.log(
                                {
                                    f"{k}_{k_i}": v_i,
                                    f"global_step": engine.state.iteration,
                                }
                            )
                    else:
                        self.writer.log(
                            {f"{k}": v, f"global_step": engine.state.iteration}
                        )
            if "text" in self.cfg.logger_name and idist.get_rank() == 0:
                log_dict = {
                    **engine.state.output["losses"],
                    f"global_step": engine.state.iteration,
                }
                pprint.pprint(log_dict)
                ##
                with open(self.log_file_path, "a") as f:
                    f.write(f"[ITER {engine.state.iteration}] ")
                    pprint.pprint(log_dict, stream=f)
                    f.write("\n")
                ##

                if "tensorboard" in self.cfg.logger_name and idist.get_rank() == 0:
                    for key, value in engine.state.output["losses"].items():
                        if isinstance(value, dict):
                            for k_i, v_i in value.items():
                                self.tensor_writer.add_scalar(f"{key}_{k_i}", v_i, engine.state.iteration)
                        else:
                            self.tensor_writer.add_scalar(key, value, engine.state.iteration)

        @trainer.on(Events.ITERATION_COMPLETED(every=self.cfg.log_every))
        def log_gradients(engine):
            if idist.get_rank() == 0:
                batch_norm = []
                for param in model.parameters():
                    if param.requires_grad == True:
                        if param.grad is not None:
                            batch_norm.append(param.grad.float().cpu().numpy().max())
                gradients = np.mean(batch_norm)
            if (
                "wandb" in self.cfg.logger_name
                and idist.get_rank() == 0
                and scaler is not None
            ):
                self.writer.log(
                    {
                        f"gradients": gradients,
                        f"scaler": scaler.get_scale(),
                        f"global_step": engine.state.iteration,
                    }
                )
            elif "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
                self.writer.log(
                    {
                        f"gradients": gradients,
                        f"global_step": engine.state.iteration,
                    }
                )
            if (
                "text" in self.cfg.logger_name
                and idist.get_rank() == 0
                and scaler is not None
            ):
                log_dict = {
                    f"gradients": gradients,
                    f"scaler": scaler.get_scale(),
                    f"global_step": engine.state.iteration,
                }
                pprint.pprint(log_dict)
                ##
                with open(self.log_file_path, "a") as f:
                    f.write(f"[GRAD {engine.state.iteration}] ")
                    pprint.pprint(log_dict, stream=f)
                    f.write("\n")
                ##
            elif "text" in self.cfg.logger_name and idist.get_rank() == 0:
                log_dict = {
                    f"gradients": gradients,
                    f"global_step": engine.state.iteration,
                }
                pprint.pprint(log_dict)
                ##
                with open(self.log_file_path, "a") as f:
                    f.write(f"[GRAD {engine.state.iteration}] ")
                    pprint.pprint(log_dict, stream=f)
                    f.write("\n")
                ##

        if (
            "watch_grad" in self.cfg
            and self.cfg.watch_grad == True
            and self.writer is not None
        ):
            self.writer.watch(model, log_freq=500)

    def on_valid_epoch_end(self, trainer, evaluator):
        @evaluator.on(Events.EPOCH_COMPLETED(every=1))
        def print_eval_output(engine):
            if "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
                for k, v in engine.state.metrics.items():
                    if "summary" in k:
                        continue
                    if type(v) == dict:
                        for k_i, v_i in v.items():
                            self.writer.log(
                                {f"{k}_{k_i}": v_i, f"epoch": trainer.state.epoch}
                            )
                    else:
                        self.writer.log({f"{k}": v, f"epoch": trainer.state.epoch})
            if "discord" in self.cfg.logger_name and idist.get_rank() == 0:
                results_train = []
                for k, v in collections.OrderedDict(
                    sorted(trainer.state.metrics.items())
                ).items():
                    if "summary" in k:
                        continue
                    if type(v) != dict:
                        results_train.append({"name": k, "value": v, "inline": True})

                results_valid = []
                for k, v in collections.OrderedDict(
                    sorted(engine.state.metrics.items())
                ).items():
                    if "summary" in k:
                        continue
                    if type(v) != dict:
                        results_valid.append({"name": k, "value": v, "inline": True})

                results = []
                for rt, rv in zip(results_train, results_valid):
                    name = rt["name"].replace("train_", "").replace("valid_", "")
                    rtv = "{:.4f}".format(round(rt["value"], 4))
                    rvv = "{:.4f}".format(round(rv["value"], 4))
                    results.append(
                        {
                            "name": name,
                            "value": f"{rtv} / {rvv}",
                            "inline": True,
                        }
                    )
                self.discord_hook.send_message(
                    content=None,
                    description=f"Results for Epoch: {trainer.state.epoch}",
                    results=results,
                    img=None,
                    color=16711680,
                )
            if "text" in self.cfg.logger_name and idist.get_rank() == 0:
                dict_res = {}
                for k, v in engine.state.metrics.items():
                    if "summary" in k:
                        continue
                    if type(v) == dict:
                        for k_i, v_i in v.items():
                            dict_res[f"{k}_{k_i}"] = v_i
                    else:
                        dict_res[f"{k}"] = v

                pprint.pprint({"epoch": trainer.state.epoch, **dict_res})
                ##
                with open(self.log_file_path, "a") as f:
                    f.write(f"\n[VALID EPOCH {trainer.state.epoch}]\n")
                    pprint.pprint({"epoch": trainer.state.epoch, **dict_res}, stream=f)
                    f.write("\n")
                ##

                if "tensorboard" in self.cfg.logger_name and idist.get_rank() == 0:
                    for key, value in dict_res.items():
                        self.tensor_writer.add_scalar(key, value, trainer.state.epoch)


    def on_completion(self, trainer):
        @trainer.on(Events.COMPLETED)
        def finish_logging(engine):
            if "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
                self.writer.finish()
            ##
            with open(self.log_file_path, "a") as f:
                f.write("\n=== TRAINING COMPLETED ===\n")
            ##
