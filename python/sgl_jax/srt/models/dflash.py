import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.speculative.dflash_utils import resolve_dflash_mask_token_id
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import AttentionType, RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def _get_dflash_config(cfg: Any) -> dict:
    dflash_cfg = getattr(cfg, "dflash_config", None)
    if dflash_cfg is None:
        return {}
    if isinstance(dflash_cfg, dict):
        return dflash_cfg
    try:
        return dict(dflash_cfg)
    except Exception:
        return {}


class DFlashAttention(nnx.Module):
    """Draft attention (encoder-only) matching the DFLASH PR semantics.

    Note: SGLang-JAX FlashAttention backend uses `custom_mask` to switch off causal
    masking. The DFLASH worker must provide a non-empty `custom_mask` for draft
    blocks, otherwise attention will remain causal.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
        layer_id: int,
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.layer_id = int(layer_id)

        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
        head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))

        rope_theta = float(getattr(config, "rope_theta", 1000000.0))
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.hidden_size = hidden_size
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.q_norm = RMSNorm(head_dim, epsilon=rms_norm_eps, param_dtype=dtype)
        self.k_norm = RMSNorm(head_dim, epsilon=rms_norm_eps, param_dtype=dtype)

        self.rotary_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
            dtype=dtype,
        )

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=self.layer_id,
            sliding_window_size=0,
            attn_type=AttentionType.ENCODER_ONLY,
        )

    def __call__(
        self,
        *,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(-1, self.q_head_num, self.head_dim)
        k = k.reshape(-1, self.kv_head_num, self.head_dim)
        v = v.reshape(-1, self.kv_head_num, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        attn_out, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        out, _ = self.o_proj(attn_out)
        return out, kv_fused

    def kv_proj_only(self, *, hidden_states: jax.Array) -> tuple[jax.Array, jax.Array]:
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        k = k.reshape(-1, self.kv_head_num, self.head_dim)
        v = v.reshape(-1, self.kv_head_num, self.head_dim)
        k = self.k_norm(k)
        return k, v

    def apply_k_rope(self, *, positions: jax.Array, k: jax.Array) -> jax.Array:
        dummy_q = jnp.zeros((k.shape[0], 1, self.head_dim), dtype=k.dtype)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DFlashMLP(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
        layer_id: int,
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.layer_id = int(layer_id)

        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(f"Invalid intermediate_size={intermediate_size} for DFLASH.")

        dflash_cfg = _get_dflash_config(config)
        mlp_bias = bool(dflash_cfg.get("mlp_bias", False))

        self.gate_up_proj = LinearBase(
            input_size=hidden_size,
            output_size=2 * intermediate_size,
            use_bias=mlp_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            use_bias=mlp_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        x = jax.nn.silu(gate) * up
        x, _ = self.down_proj(x)
        return x


class DFlashDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
        layer_id: int,
    ) -> None:
        self.layer_id = int(layer_id)
        self.mesh = mesh
        self.dtype = dtype

        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, epsilon=rms_norm_eps, param_dtype=dtype)
        self.self_attn = DFlashAttention(config, mesh=mesh, dtype=dtype, layer_id=layer_id)
        self.post_attention_layernorm = RMSNorm(hidden_size, epsilon=rms_norm_eps, param_dtype=dtype)
        self.mlp = DFlashMLP(config, mesh=mesh, dtype=dtype, layer_id=layer_id)

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        attn_out, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        hidden_states = attn_out + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, kv_fused


class DFlashDraftModel(nnx.Module):
    """DFLASH draft model for SGLang-JAX (TPU).

    This keeps a token embedding + LM head so we can run it inside the current
    SGLang-JAX runtime (which passes `input_ids`, not `input_embeds`).

    NOTE: DFLASH draft checkpoints often omit embed/head weights (they are shared
    from the target model). In sglang-jax we always share them at runtime via
    `set_embed_and_head()`, and the weight loader intentionally does NOT require
    embed/head tensors to exist in the draft checkpoint.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.dtype = dtype

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        dflash_cfg = _get_dflash_config(config)
        self.mask_token_id = resolve_dflash_mask_token_id(draft_hf_config=config)
        self.use_mask_embedding = bool(dflash_cfg.get("use_mask_embedding", False))
        # Learned embedding used for masked positions (when enabled). This
        # matches our EasyDeL TPU training checkpoints, which store the vector
        # under `model/mask_embedding/value`.
        self.mask_embedding = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (hidden_size,),
                dtype=dtype,
                out_sharding=jax.sharding.PartitionSpec(None),
            )
        )

        target_layer_ids = dflash_cfg.get("target_layer_ids", None)
        if target_layer_ids is None:
            self.num_context_features = num_layers
        else:
            self.num_context_features = len(list(target_layer_ids))

        block_size = dflash_cfg.get("block_size", None)
        if block_size is None:
            block_size = getattr(config, "block_size", 16)
        self.block_size = int(block_size)

        self.embed_tokens = Embed(
            num_embeddings=int(config.vocab_size),
            features=hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )
        self.layers = nnx.data(
            [DFlashDecoderLayer(config, mesh=mesh, dtype=dtype, layer_id=i) for i in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_size, epsilon=rms_norm_eps, param_dtype=dtype)

        fc_bias = bool(dflash_cfg.get("fc_bias", False))
        self.fc = LinearBase(
            input_size=self.num_context_features * hidden_size,
            output_size=hidden_size,
            use_bias=fc_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.hidden_norm = RMSNorm(hidden_size, epsilon=rms_norm_eps, param_dtype=dtype)

        self.lm_head = ParallelLMHead(
            int(config.vocab_size),
            hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
        )
        self.logits_processor = LogitsProcessor(int(config.vocab_size), mesh=self.mesh)

    def set_embed_and_head(self, embed_weight: jax.Array | None, head_weight: jax.Array | None) -> None:
        if embed_weight is not None:
            self.embed_tokens.embedding.value = embed_weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def get_embed_and_head(self):
        return (self.embed_tokens.embedding.value, self.lm_head.embedding.value)

    def project_target_hidden(self, target_hidden: jax.Array) -> jax.Array:
        x, _ = self.fc(target_hidden)
        return self.hidden_norm(x)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        if self.use_mask_embedding:
            mask = forward_batch.input_ids == int(self.mask_token_id)
            hidden_states = jnp.where(
                mask[:, None],
                self.mask_embedding.value.astype(hidden_states.dtype),
                hidden_states,
            )
        residual = None
        layers_kv_fused = []
        layers_callback_flag = []

        for layer in self.layers:
            hidden_states, residual, kv_fused = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)

        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)

        output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag

    def load_weights(self, model_config):
        loader = WeightLoader(model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype)
        loader.load_weights_from_safetensors(self._create_weight_mappings())
        logger.info("DFLASH draft weights loaded successfully!")

    def _create_weight_mappings(self) -> dict[str, WeightMapping]:
        dflash_cfg = _get_dflash_config(self.config)
        fc_bias = bool(dflash_cfg.get("fc_bias", False))
        mlp_bias = bool(dflash_cfg.get("mlp_bias", False))

        mappings: dict[str, WeightMapping] = {
            "model.norm.weight": WeightMapping(
                target_path="norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            "model.mask_embedding": WeightMapping(
                target_path="mask_embedding",
                sharding=(None,),
                transpose=False,
            ),
            # DFLASH projection weights
            "model.fc.weight": WeightMapping(
                target_path="fc.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            "model.hidden_norm.weight": WeightMapping(
                target_path="hidden_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }
        if fc_bias:
            mappings["model.fc.bias"] = WeightMapping(
                target_path="fc.bias",
                sharding=("tensor",),
                transpose=False,
            )

        for i in range(int(self.config.num_hidden_layers)):
            prefix = f"model.layers.{i}"
            target_prefix = f"layers.{i}"
            mappings.update(
                {
                    f"{prefix}.input_layernorm.weight": WeightMapping(
                        target_path=f"{target_prefix}.input_layernorm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                        target_path=f"{target_prefix}.post_attention_layernorm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.q_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.k_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.v_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.o_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.q_norm.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.q_norm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.k_norm.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.k_norm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.mlp.gate_up_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.mlp.gate_up_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.mlp.down_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.mlp.down_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                    ),
                }
            )
            if mlp_bias:
                mappings.update(
                    {
                        f"{prefix}.mlp.gate_up_proj.bias": WeightMapping(
                            target_path=f"{target_prefix}.mlp.gate_up_proj.bias",
                            sharding=("tensor",),
                            transpose=False,
                        ),
                        f"{prefix}.mlp.down_proj.bias": WeightMapping(
                            target_path=f"{target_prefix}.mlp.down_proj.bias",
                            sharding=(None,),
                            transpose=False,
                        ),
                    }
                )

            if bool(getattr(self.config, "attention_bias", False)):
                mappings.update(
                    {
                        f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                            target_path=f"{target_prefix}.self_attn.q_proj.bias",
                            sharding=("tensor",),
                            transpose=False,
                        ),
                        f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                            target_path=f"{target_prefix}.self_attn.k_proj.bias",
                            sharding=("tensor",),
                            transpose=False,
                        ),
                        f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                            target_path=f"{target_prefix}.self_attn.v_proj.bias",
                            sharding=("tensor",),
                            transpose=False,
                        ),
                        f"{prefix}.self_attn.o_proj.bias": WeightMapping(
                            target_path=f"{target_prefix}.self_attn.o_proj.bias",
                            sharding=(None,),
                            transpose=False,
                        ),
                    }
                )

        return mappings


EntryClass = DFlashDraftModel
