import json
import math

import pytest
import torch

from llm import generate
from llm.checkpoint import BEST, LAST, load_checkpoint, load_model
from llm.config import Config, TrainConfig
from llm.prepare import prepare
from llm.tokenizer import BPETokenizer
from llm.train import lr_at, train


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    text = "".join(f"line {i}: the cat sat on the mat and the dog ran in the fog.\n" for i in range(600))
    (root / "input.txt").write_text(text, encoding="utf-8")
    meta = prepare([root / "input.txt"], root / "out", vocab_size=300)
    assert meta["train_tokens"] > meta["val_tokens"] > 0
    return root / "out"


def make_config(data_dir, out_dir, max_steps):
    return Config.from_dict(
        {
            "model": {"block_size": 32, "n_layer": 2, "n_head": 2, "n_embd": 32, "dropout": 0.0},
            "data": {"dir": str(data_dir)},
            "train": {
                "out_dir": str(out_dir),
                "device": "cpu",
                "batch_size": 8,
                "max_steps": max_steps,
                "learning_rate": 3e-3,
                "warmup_steps": 5,
                "eval_interval": 20,
                "eval_iters": 4,
                "log_interval": 10,
            },
        }
    )


def test_prepare_outputs_roundtrip(data_dir):
    meta = json.loads((data_dir / "meta.json").read_text())
    tok = BPETokenizer.load(data_dir / "tokenizer.json")
    assert meta["vocab_size"] == tok.vocab_size
    assert (data_dir / "train.bin").stat().st_size == meta["train_tokens"] * 2  # uint16
    assert (data_dir / "val.bin").stat().st_size == meta["val_tokens"] * 2


def test_train_resume_and_generate(data_dir, tmp_path, capsys):
    out_dir = tmp_path / "run"
    result = train(make_config(data_dir, out_dir, max_steps=40))
    assert result["step"] == 40
    assert result["val_loss"] < math.log(300) - 1.0  # well below the uniform baseline
    assert (out_dir / LAST).exists() and (out_dir / BEST).exists()
    assert load_checkpoint(out_dir / LAST)["step"] == 40

    resumed = train(make_config(data_dir, out_dir, max_steps=60), resume=True)
    assert resumed["step"] == 60
    assert resumed["val_loss"] < result["val_loss"]

    rows = [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows if "eval_val_loss" in r] == [20, 40, 60]

    model, cfg = load_model(out_dir / BEST)
    assert not model.training
    assert cfg.model.vocab_size == 300

    generate.main(["--run", str(out_dir), "--prompt", "the cat", "--max-new-tokens", "8", "--device", "cpu"])
    assert capsys.readouterr().out.startswith("the cat")


def test_resume_rejects_a_different_architecture(data_dir, tmp_path):
    out_dir = tmp_path / "run"
    train(make_config(data_dir, out_dir, max_steps=20))
    cfg = make_config(data_dir, out_dir, max_steps=40)
    cfg.model.n_embd = 64
    with pytest.raises(ValueError, match="model config differs"):
        train(cfg, resume=True)


def test_lr_schedule():
    cfg = TrainConfig(learning_rate=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    assert lr_at(0, cfg) == pytest.approx(0.1)
    assert lr_at(9, cfg) == pytest.approx(1.0)
    assert lr_at(10, cfg) == pytest.approx(1.0)
    assert lr_at(60, cfg) == pytest.approx(0.55)
    assert lr_at(110, cfg) == pytest.approx(0.1)
    assert all(lr_at(s, cfg) >= lr_at(s + 1, cfg) for s in range(10, 110))


def test_config_loading(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  n_layer: 3\ntrain:\n  learning_rate: 3e-4\n  max_steps: 10\n")
    cfg = Config.load(path, ["train.max_steps=25", "model.dropout=0"])
    assert cfg.model.n_layer == 3
    assert cfg.train.learning_rate == pytest.approx(3e-4)  # PyYAML reads 3e-4 as a string
    assert cfg.train.max_steps == 25
    assert cfg.model.dropout == 0.0
    assert Config.from_dict(cfg.to_dict()) == cfg

    with pytest.raises(ValueError, match="unknown keys"):
        Config.from_dict({"train": {"learning_rat": 1e-3}})
    with pytest.raises(ValueError, match="expected int"):
        Config.from_dict({"train": {"max_steps": "many"}})


def test_checkpoints_load_with_weights_only(data_dir, tmp_path):
    out_dir = tmp_path / "run"
    train(make_config(data_dir, out_dir, max_steps=20))
    ckpt = torch.load(out_dir / LAST, weights_only=True)
    assert set(ckpt) == {"model", "optimizer", "scaler", "step", "best_val_loss", "config"}
