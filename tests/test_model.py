import math

import pytest
import torch

from llm.model import GPT, GPTConfig, _sample


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=65, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)).eval()


def test_forward_shapes_and_initial_loss(model):
    idx = torch.randint(65, (4, 16))
    # Independent targets: with tied embeddings, predicting the input token itself is easier.
    logits, loss = model(idx, torch.randint(65, (4, 16)))
    assert logits.shape == (4, 16, 65)
    # An untrained model should be close to uniform over the vocabulary.
    assert abs(loss.item() - math.log(65)) < 0.3
    assert model(idx)[1] is None


def test_weights_are_tied(model):
    assert model.lm_head.weight is model.wte.weight


def test_rejects_sequences_longer_than_block_size(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_causality(model):
    a = torch.randint(65, (1, 16))
    b = a.clone()
    b[0, 10:] = (b[0, 10:] + 1) % 65
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-5)
    assert not torch.allclose(la[0, 10:], lb[0, 10:], atol=1e-5)


def test_kv_cache_matches_full_forward(model):
    idx = torch.randint(65, (2, 12))
    full, _ = model._hidden(idx)
    h1, past = model._hidden(idx[:, :5])
    h2, past = model._hidden(idx[:, 5:9], past)  # several new tokens at once
    h3, _ = model._hidden(idx[:, 9:10], past)  # a single new token
    assert torch.allclose(torch.cat((h1, h2, h3), dim=1), full[:, :10], atol=1e-5)


def test_greedy_generate_matches_uncached_loop(model):
    prompt = torch.randint(65, (2, 4))
    n = 20  # runs past block_size, exercising the sliding window
    out = model.generate(prompt, n, temperature=0)
    assert out.shape == (2, 4 + n)

    ref = prompt
    with torch.no_grad():
        for _ in range(n):
            logits, _ = model(ref[:, -16:])
            ref = torch.cat((ref, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    assert torch.equal(out, ref)


def test_generate_restores_training_mode(model):
    model.train()
    model.generate(torch.zeros(1, 1, dtype=torch.long), 2)
    assert model.training


def test_generate_stops_at_eos(model):
    prompt = torch.randint(65, (1, 4))
    eos = model.generate(prompt, 1, temperature=0)[0, -1].item()
    out = model.generate(prompt, 10, temperature=0, eos_id=eos)
    assert out.shape == (1, 5)


def test_sampling_filters():
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0]])
    assert _sample(logits, 0, None, None).item() == 1
    for _ in range(20):
        assert _sample(logits, 1.0, 1, None).item() == 1
        assert _sample(logits, 1.0, None, 0.01).item() == 1
        assert _sample(logits, 1.0, 2, None).item() in (1, 3)


def test_optimizer_decays_only_matrices(model):
    opt = model.configure_optimizer(1e-3, 0.1, (0.9, 0.95))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == model.num_params()


def test_overfits_a_single_batch(model):
    model.train()
    x = torch.randint(65, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = model.configure_optimizer(3e-3, 0.0, (0.9, 0.95))
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
