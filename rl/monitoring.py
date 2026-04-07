from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import csv
import json
import time
from datetime import datetime


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    root: Path
    logs_dir: Path
    eval_video_dir: Path
    replay_dir: Path
    tensorboard_dir: Path
    metrics_csv_path: Path


def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def prepare_artifacts(output_root: str, run_id: str) -> RunArtifacts:
    root = Path(output_root)
    logs_dir = root / "logs" / run_id
    eval_video_dir = root / "eval_videos" / run_id
    replay_dir = root / "replays" / run_id
    tensorboard_dir = root / "tensorboard" / run_id

    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_video_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv_path = logs_dir / "metrics.csv"
    return RunArtifacts(
        run_id=run_id,
        root=root,
        logs_dir=logs_dir,
        eval_video_dir=eval_video_dir,
        replay_dir=replay_dir,
        tensorboard_dir=tensorboard_dir,
        metrics_csv_path=metrics_csv_path,
    )


class CSVMetricLogger:
    _FLUSH_INTERVAL = 5.0  # seconds between forced flushes

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._fieldnames = [
            "timestamp",
            "run_id",
            "row_type",
            "rollout",
            "train_team",
            "total_steps",
            "rollout_steps",
            "completed_episodes_in_rollout",
            "episode_index",
            "episode_length",
            "ep_h_return",
            "ep_s_return",
            "agent_return_0",
            "agent_return_1",
            "agent_return_2",
            "agent_return_3",
            "agent_return_4",
            "agent_return_5",
            "h_return_last10",
            "s_return_last10",
            "lr_h",
            "lr_s",
            "h_pg_loss",
            "h_vf_loss",
            "h_entropy",
            "s_pg_loss",
            "s_vf_loss",
            "s_entropy",
            "eval_h_return",
            "eval_s_return",
            "eval_ep_length",
            "eval_hiders_caught",
            "eval_video",
            "eval_replay",
            "eval_error",
        ]
        self._fp = self.csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=self._fieldnames)
        if self.csv_path.stat().st_size == 0:
            self._writer.writeheader()
            self._fp.flush()
        self._last_flush = time.monotonic()

    def log(self, row: Dict[str, Any]) -> None:
        payload = {k: row.get(k, "") for k in self._fieldnames}
        self._writer.writerow(payload)
        # P2 fix: buffered flush — only flush periodically, not every row.
        now = time.monotonic()
        if now - self._last_flush >= self._FLUSH_INTERVAL:
            self._fp.flush()
            self._last_flush = now

    def close(self) -> None:
        self._fp.flush()
        self._fp.close()


class TensorboardLogger:
    def __init__(self, log_dir: Path):
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(log_dir))
        except Exception:
            self._writer = None

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is None:
            return
        self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def save_replay_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
