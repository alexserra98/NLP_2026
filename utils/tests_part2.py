import torch as t
import torch.nn as nn
import sys
from dataclasses import dataclass
from pathlib import Path
from functools import partial

exercises_dir = Path(__file__).parent.parent
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


# ---------------------------------------------------------------------------
# Internal reference implementations used only by the tests below.
# These are intentionally minimal -- just enough to produce ground-truth outputs.
# Prefixed with _ to signal they are test-internal and not part of the exercise.
# ---------------------------------------------------------------------------

_device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")


@dataclass
class _Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


class _LayerNorm(nn.Module):
    """Reference LayerNorm used in test_layer_norm_epsilon."""

    def __init__(self, cfg: _Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual):
        mean = residual.mean(dim=-1, keepdim=True)
        std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()
        return (residual - mean) / std * self.w + self.b


class _CausalMaskReference(nn.Module):
    """Reference causal mask used in test_causal_mask.
    Provides the IGNORE buffer and a known-correct apply_causal_mask."""

    def __init__(self, cfg: _Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32))

    def apply_causal_mask(self, attn_scores):
        all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        mask = t.triu(all_ones, diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


# ---------------------------------------------------------------------------
# Legacy test functions (kept for backwards compatibility)
# ---------------------------------------------------------------------------


def rand_float_test(cls, shape):
    cfg = _Config(debug=True)
    layer = cls(cfg).to(_device)
    random_input = t.randn(shape).to(_device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def rand_int_test(cls, shape):
    cfg = _Config(debug=True)
    layer = cls(cfg).to(_device)
    random_input = t.randint(100, 1000, shape).to(_device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def load_gpt2_test(cls, gpt2_layer, input):
    cfg = _Config(debug=True)
    layer = cls(cfg).to(_device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    orig_input = input.clone()
    output = layer(orig_input)
    assert t.allclose(input, orig_input), "Input has been modified, make sure operations are not done in place"
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape)
    try:
        reference_output = gpt2_layer(input)
    except:
        reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum() / comparison.numel():.2%} of the values are correct\n")
    assert 1 - (comparison.sum() / comparison.numel()) < 1e-5, "More than 0.01% of the values are incorrect"


def test_layer_norm_epsilon(layer_norm, cache):
    cfg = _Config(layer_norm_eps=0.1)
    expected = _LayerNorm(cfg).to(_device)(cache.clone())
    actual = layer_norm(cfg).to(_device)(cache.clone())
    t.testing.assert_close(expected, actual, msg="LayerNorm: Is your epsilon inside the sqrt?")


def test_causal_mask(apply_causal_mask):
    cfg = _Config()
    attn_scores = t.randn((1, 1, 5, 5)).to(_device)
    ref = _CausalMaskReference(cfg).to(_device)

    sol_apply_causal_mask = ref.apply_causal_mask
    apply_causal_mask = partial(apply_causal_mask, self=ref)

    expected = sol_apply_causal_mask(attn_scores=attn_scores.clone())
    actual = apply_causal_mask(attn_scores=attn_scores.clone())

    def print_scores():
        print(f"Expected Attention Scores: \n {expected}")
        print(f"Actual Attention Scores: \n {actual}")
        print(f"Expected Attention Probs: \n {t.softmax(expected, dim=-1)}")
        print(f"Actual Attention Probs: \n {t.softmax(actual, dim=-1)}")

    if t.any(t.isnan(actual)):
        nan_freq = t.sum(t.isnan(actual)).item() / actual.numel()
        print_scores()
        raise ValueError(
            f"Your masked attention scores contains {nan_freq * 100}% NaNs. "
            "Make sure you aren't multiplying 0 * neg inf anywhere."
        )

    attn_probs = t.softmax(actual, dim=-1)

    if t.any(t.isnan(attn_probs)):
        print_scores()
        nan_freq = t.sum(t.isnan(attn_probs)).item() / attn_probs.numel()
        raise ValueError(
            f"Your post-softmax masked attention scores contains {nan_freq * 100}% NaNs. "
            "Make sure you aren't setting an entire row to neg inf."
        )

    if not t.allclose(actual, expected):
        print_scores()
        t.testing.assert_close(actual, expected)

    print("All tests in `test_causal_mask` passed!")


# ---------------------------------------------------------------------------
# Consolidated exercise tests
# Each test_* function is a single entry point for one exercise.
# It runs shape checks, property checks, and (where possible) GPT-2 comparison.
# ---------------------------------------------------------------------------


def test_embed(Embed, gpt2_embed=None, tokens=None):
    """Tests for the Embed module."""
    cfg = _Config(debug=True, d_vocab=100, d_model=16)
    layer = Embed(cfg).to(_device)

    # 1. Output shape
    tok = t.tensor([[1, 2, 3], [4, 5, 6]]).to(_device)
    output = layer(tok)
    assert output.shape == (2, 3, 16), (
        f"Expected output shape (2, 3, 16), got {tuple(output.shape)}"
    )

    # 2. Same token -> same embedding
    tok_repeat = t.tensor([[7, 7, 7]]).to(_device)
    out = layer(tok_repeat)
    assert t.allclose(out[0, 0], out[0, 1]) and t.allclose(out[0, 1], out[0, 2]), (
        "The same token ID should produce the same embedding vector"
    )

    # 3. Single token batch
    single = t.tensor([[42]]).to(_device)
    out_single = layer(single)
    assert out_single.shape == (1, 1, 16), (
        f"Expected shape (1, 1, 16) for single token, got {tuple(out_single.shape)}"
    )

    # 4. Different tokens should give different embeddings
    tok_diff = t.tensor([[0, 1]]).to(_device)
    out_diff = layer(tok_diff)
    assert not t.allclose(out_diff[0, 0], out_diff[0, 1], atol=1e-4), (
        "Different token IDs should produce different embeddings"
    )

    # 5. GPT-2 comparison (if reference layer provided)
    if gpt2_embed is not None and tokens is not None:
        load_gpt2_test(Embed, gpt2_embed, tokens)

    print("All tests in `test_embed` passed!")


def test_pos_embed(PosEmbed, gpt2_pos_embed=None, tokens=None):
    """Tests for the PosEmbed module."""
    cfg = _Config(debug=True, d_vocab=100, d_model=16, n_ctx=32)
    layer = PosEmbed(cfg).to(_device)

    # 1. Output shape
    tok = t.randint(0, 100, (2, 5)).to(_device)
    output = layer(tok)
    assert output.shape == (2, 5, 16), (
        f"Expected output shape (2, 5, 16), got {tuple(output.shape)}"
    )

    # 2. Position embeddings should NOT depend on token identity
    tok_a = t.tensor([[10, 20, 30]]).to(_device)
    tok_b = t.tensor([[99, 1, 50]]).to(_device)
    out_a = layer(tok_a)
    out_b = layer(tok_b)
    assert t.allclose(out_a, out_b), (
        "Positional embeddings should be the same regardless of token content "
        "(they depend only on position, not on what token is at that position)"
    )

    # 3. Different positions should have different embeddings
    tok_seq = t.tensor([[0, 1, 2, 3]]).to(_device)
    out = layer(tok_seq)
    assert not t.allclose(out[0, 0], out[0, 1], atol=1e-4), (
        "Different positions should (almost certainly) have different embeddings"
    )

    # 4. Batch dimension: same positions across batch should match
    tok_batch = t.randint(0, 100, (3, 4)).to(_device)
    out_batch = layer(tok_batch)
    assert t.allclose(out_batch[0], out_batch[1]) and t.allclose(out_batch[1], out_batch[2]), (
        "Positional embeddings should be identical across all items in the batch"
    )

    # 5. GPT-2 comparison
    if gpt2_pos_embed is not None and tokens is not None:
        load_gpt2_test(PosEmbed, gpt2_pos_embed, tokens)

    print("All tests in `test_pos_embed` passed!")


def test_mlp(MLP):
    """Tests for the MLP module."""
    cfg = _Config(debug=True, d_model=16, d_mlp=64)
    layer = MLP(cfg).to(_device)

    # 1. Output shape must match input shape
    x = t.randn(2, 5, 16).to(_device)
    output = layer(x)
    assert output.shape == (2, 5, 16), (
        f"Expected output shape (2, 5, 16), got {tuple(output.shape)}"
    )

    # 2. Zero input should NOT give zero output (because of biases)
    zero_input = t.zeros(1, 1, 16).to(_device)
    zero_output = layer(zero_input)
    assert not t.allclose(zero_output, t.zeros_like(zero_output), atol=1e-6), (
        "MLP with zero input should produce non-zero output (nn.Linear has biases)"
    )

    # 3. Deterministic: same input twice -> same output
    x = t.randn(1, 3, 16).to(_device)
    out1 = layer(x.clone())
    out2 = layer(x.clone())
    assert t.allclose(out1, out2), (
        "MLP should be deterministic: same input should always produce same output"
    )

    # 4. Verify it uses GELU (not ReLU): negative inputs should still vary the output
    x_neg = -t.ones(1, 1, 16).to(_device)
    x_more_neg = -2 * t.ones(1, 1, 16).to(_device)
    out_neg = layer(x_neg)
    out_more_neg = layer(x_more_neg)
    assert not t.allclose(out_neg, out_more_neg, atol=1e-5), (
        "MLP output should vary with different negative inputs (check your activation function)"
    )

    print("All tests in `test_mlp` passed!")


def test_causal_mask_all(apply_causal_mask):
    """Combined causal mask test: reference comparison + detailed property checks."""
    cfg = _Config()
    ref = _CausalMaskReference(cfg).to(_device)

    # --- Part 1: Reference comparison (from test_causal_mask) ---
    attn_scores = t.randn((1, 1, 5, 5)).to(_device)
    apply_fn = partial(apply_causal_mask, self=ref)

    expected = ref.apply_causal_mask(attn_scores=attn_scores.clone())
    actual = apply_fn(attn_scores=attn_scores.clone())

    if t.any(t.isnan(actual)):
        nan_freq = t.sum(t.isnan(actual)).item() / actual.numel()
        raise ValueError(
            f"Your masked attention scores contains {nan_freq * 100}% NaNs. "
            "Make sure you aren't multiplying 0 * neg inf anywhere."
        )

    attn_probs = t.softmax(actual, dim=-1)
    if t.any(t.isnan(attn_probs)):
        nan_freq = t.sum(t.isnan(attn_probs)).item() / attn_probs.numel()
        raise ValueError(
            f"Your post-softmax attention probs contain {nan_freq * 100}% NaNs. "
            "Make sure you aren't setting an entire row to neg inf."
        )

    if not t.allclose(actual, expected):
        t.testing.assert_close(actual, expected)

    # --- Part 2: Detailed property checks ---
    scores = t.zeros(1, 1, 4, 4).to(_device)
    masked = apply_fn(attn_scores=scores.clone())

    # Upper triangle should be -inf, lower triangle + diagonal should not
    for q in range(4):
        for k in range(4):
            if k > q:
                assert masked[0, 0, q, k] == float("-inf"), (
                    f"Position [{q},{k}] should be -inf (future position), got {masked[0, 0, q, k]:.2f}"
                )
            else:
                assert masked[0, 0, q, k] != float("-inf"), (
                    f"Position [{q},{k}] should NOT be masked (current or past position)"
                )

    # After softmax, each row sums to 1
    probs = t.softmax(masked, dim=-1)
    row_sums = probs.sum(dim=-1)
    assert t.allclose(row_sums, t.ones_like(row_sums), atol=1e-5), (
        "After softmax, each row of attention probabilities should sum to 1"
    )

    # After softmax, future positions should have probability 0
    for q in range(4):
        for k in range(q + 1, 4):
            assert probs[0, 0, q, k] < 1e-6, (
                f"After softmax, position [{q},{k}] should be ~0 (future), got {probs[0, 0, q, k]:.6f}"
            )

    # Works with multiple heads and batches
    scores_mh = t.randn(2, 4, 6, 6).to(_device)
    masked_mh = apply_fn(attn_scores=scores_mh.clone())
    assert masked_mh.shape == (2, 4, 6, 6), (
        f"Output shape should match input shape, got {tuple(masked_mh.shape)}"
    )
    for b in range(2):
        for h in range(4):
            for q in range(6):
                for k in range(q + 1, 6):
                    assert masked_mh[b, h, q, k] == float("-inf"), (
                        f"Position [batch={b}, head={h}, {q}, {k}] should be -inf"
                    )

    print("All tests in `test_causal_mask_all` passed!")


def test_attention(Attention):
    """Tests for the Attention module."""
    cfg = _Config(debug=True, d_model=32, d_head=8, n_heads=4, n_ctx=16)
    layer = Attention(cfg).to(_device)

    # 1. Output shape: should be (batch, seq, d_model)
    x = t.randn(2, 5, 32).to(_device)
    output = layer(x)
    assert output.shape == (2, 5, 32), (
        f"Expected output shape (2, 5, 32), got {tuple(output.shape)}"
    )

    # 2. Deterministic
    x = t.randn(1, 4, 32).to(_device)
    out1 = layer(x.clone())
    out2 = layer(x.clone())
    assert t.allclose(out1, out2), (
        "Attention should be deterministic: same input should produce same output"
    )

    # 3. Causality: changing a future token should not affect output at earlier positions
    x_base = t.randn(1, 6, 32).to(_device)
    x_modified = x_base.clone()
    x_modified[0, 4, :] += 10.0  # perturb position 4
    out_base = layer(x_base)
    out_modified = layer(x_modified)
    # Positions 0-3 should be unchanged
    assert t.allclose(out_base[0, :4], out_modified[0, :4], atol=1e-5), (
        "Changing a future token should not affect output at earlier positions (causal violation). "
        "Make sure you're applying the causal mask correctly."
    )
    # Position 4 should change (the perturbed position itself sees its own input)
    assert not t.allclose(out_base[0, 4], out_modified[0, 4], atol=1e-4), (
        "Changing a token should affect the output at that same position"
    )

    # 4. Single token: should work without errors
    x_single = t.randn(1, 1, 32).to(_device)
    out_single = layer(x_single)
    assert out_single.shape == (1, 1, 32), (
        f"Expected shape (1, 1, 32) for single token, got {tuple(out_single.shape)}"
    )

    print("All tests in `test_attention` passed!")


def test_transformer_block(TransformerBlock):
    """Tests for the TransformerBlock module."""
    cfg = _Config(debug=True, d_model=32, d_head=8, n_heads=4, d_mlp=128, n_ctx=16)
    layer = TransformerBlock(cfg).to(_device)

    # 1. Output shape matches input shape
    x = t.randn(2, 5, 32).to(_device)
    output = layer(x)
    assert output.shape == (2, 5, 32), (
        f"Expected output shape (2, 5, 32), got {tuple(output.shape)}"
    )

    # 2. Residual connection: output should not be zero even for zero-ish input
    #    (because LayerNorm + biases produce non-zero intermediate values)
    small_input = t.ones(1, 3, 32).to(_device) * 0.001
    out_small = layer(small_input)
    assert not t.allclose(out_small, t.zeros_like(out_small), atol=1e-6), (
        "TransformerBlock should produce non-zero output (residual connections + biases)"
    )

    # 3. Residual connection check: output should be "close" to input in magnitude
    #    (the residual stream means the block adds to the input, not replaces it)
    x = t.randn(1, 4, 32).to(_device) * 5.0
    out = layer(x)
    diff_norm = (out - x).norm()
    input_norm = x.norm()
    assert diff_norm < input_norm * 10, (
        "The difference between input and output is very large relative to the input. "
        "Make sure you're adding the residual connection (input + sublayer_output), not just returning the sublayer output."
    )

    # 4. Causality inherited from attention
    x_base = t.randn(1, 6, 32).to(_device)
    x_mod = x_base.clone()
    x_mod[0, 5, :] += 10.0
    out_base = layer(x_base)
    out_mod = layer(x_mod)
    assert t.allclose(out_base[0, :5], out_mod[0, :5], atol=1e-5), (
        "TransformerBlock should be causal: changing a future token should not affect earlier positions"
    )

    print("All tests in `test_transformer_block` passed!")


def test_unembed(Unembed):
    """Tests for the Unembed module."""
    cfg = _Config(debug=True, d_model=32, d_vocab=100)
    layer = Unembed(cfg).to(_device)

    # 1. Output shape: (batch, seq, d_vocab)
    x = t.randn(2, 5, 32).to(_device)
    output = layer(x)
    assert output.shape == (2, 5, 100), (
        f"Expected output shape (2, 5, 100), got {tuple(output.shape)}"
    )

    # 2. Deterministic
    x = t.randn(1, 3, 32).to(_device)
    out1 = layer(x.clone())
    out2 = layer(x.clone())
    assert t.allclose(out1, out2), (
        "Unembed should be deterministic"
    )

    # 3. Different inputs should (almost certainly) give different outputs
    x_a = t.randn(1, 1, 32).to(_device)
    x_b = t.randn(1, 1, 32).to(_device)
    out_a = layer(x_a)
    out_b = layer(x_b)
    assert not t.allclose(out_a, out_b, atol=1e-4), (
        "Different inputs should produce different logit distributions"
    )

    # 4. Bias should be frozen (not trainable)
    for name, param in layer.named_parameters():
        if 'bias' in name:
            assert not param.requires_grad, (
                f"Bias parameter '{name}' should be frozen (requires_grad=False)"
            )

    print("All tests in `test_unembed` passed!")


def test_demo_transformer(DemoTransformer):
    """Tests for the full DemoTransformer model."""
    cfg = _Config(debug=True, d_model=32, d_head=8, n_heads=4, d_mlp=128,
                  n_ctx=16, d_vocab=100, n_layers=2)
    model = DemoTransformer(cfg).to(_device)

    # 1. Output shape: (batch, seq, d_vocab)
    tokens = t.randint(0, 100, (2, 5)).to(_device)
    output = model(tokens)
    assert output.shape == (2, 5, 100), (
        f"Expected output shape (2, 5, 100), got {tuple(output.shape)}"
    )

    # 2. Deterministic
    tokens = t.randint(0, 100, (1, 4)).to(_device)
    out1 = model(tokens)
    out2 = model(tokens)
    assert t.allclose(out1, out2), (
        "DemoTransformer should be deterministic"
    )

    # 3. Causality: changing a future token should not affect logits at earlier positions
    tokens_base = t.randint(0, 100, (1, 8)).to(_device)
    tokens_mod = tokens_base.clone()
    tokens_mod[0, 6] = (tokens_mod[0, 6] + 50) % 100  # change token at position 6
    out_base = model(tokens_base)
    out_mod = model(tokens_mod)
    assert t.allclose(out_base[0, :6], out_mod[0, :6], atol=1e-5), (
        "Changing a future token should not affect logits at earlier positions (causal violation)"
    )

    # 4. Output should contain valid logits (no NaN or Inf)
    tokens = t.randint(0, 100, (1, 10)).to(_device)
    output = model(tokens)
    assert not t.any(t.isnan(output)), "Output contains NaN values"
    assert not t.any(t.isinf(output)), "Output contains Inf values"

    # 5. Has the right number of transformer blocks
    blocks = [m for m in model.modules() if hasattr(m, 'attn') and hasattr(m, 'mlp')]
    assert len(blocks) == cfg.n_layers, (
        f"Expected {cfg.n_layers} transformer blocks, found {len(blocks)}. "
        "Make sure you're using nn.ModuleList with the right number of blocks."
    )

    print("All tests in `test_demo_transformer` passed!")
