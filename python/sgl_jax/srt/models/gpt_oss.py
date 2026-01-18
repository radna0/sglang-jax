import logging
from typing import TYPE_CHECKING, Any

import jax
from jax import shard_map
from flax import nnx
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sgl_jax.srt.configs.model_config import ModelConfig


class GptOssTopKRouter(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.dtype = dtype

        self.weight = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (self.hidden_size, self.num_experts),
                dtype=dtype,
                # Keep the router replicated across TP. Sharding the expert
                # dimension makes routing TP-dependent and also causes shape
                # mismatches during dummy-weight generation.
                out_sharding=jax.sharding.PartitionSpec(None, None),
            )
        )
        self.bias = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (self.num_experts,),
                dtype=dtype,
                out_sharding=jax.sharding.PartitionSpec(None),
            )
        )
        self.mesh = mesh

    def __call__(self, hidden_states: jax.Array) -> tuple[jax.Array, jax.Array]:
        router_logits = hidden_states.astype(self.dtype) @ self.weight.value + self.bias.value
        topk_values, topk_ids = jax.lax.top_k(router_logits, k=self.top_k)
        topk_weights = jax.nn.softmax(topk_values, axis=-1).astype(hidden_states.dtype)
        return topk_weights, topk_ids


class GptOssExperts(nnx.Module):
    """GPT‑OSS experts (inference): top‑k routed, swiglu_limit + 1.702 GLU."""

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

        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_local_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.expert_dim = int(config.intermediate_size)
        self.alpha = 1.702
        self.limit = float(getattr(config, "swiglu_limit", 7.0))

        # Mirror EPMoE's layout: build a (expert, tensor) mesh even when ep_size=1,
        # so the GMM kernels and sharding rules match other MoE models.
        self.original_mesh = mesh
        self.ep_size = int(getattr(getattr(config, "ep_size", None), "__int__", lambda: 1)())
        self.ep_size = 1 if self.ep_size <= 0 else self.ep_size
        world_size = self.original_mesh.shape.get("data", 1) * self.original_mesh.shape.get("tensor", 1)
        self.tp_size = world_size // self.ep_size
        self.experts_per_device = self.num_experts // self.ep_size

        devices = self.original_mesh.devices.flatten()
        self.moe_mesh = jax.sharding.Mesh(
            devices.reshape(self.ep_size, self.tp_size),
            axis_names=("expert", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
        )
        abstract_mesh = self.original_mesh.abstract_mesh
        self.updated_mesh = abstract_mesh.update(
            axis_sizes=(self.ep_size, self.tp_size),
            axis_names=("expert", "tensor"),
        )

        # HF layout:
        # - gate_up_proj: [E, hidden, 2 * expert_dim] (even=gate, odd=up)
        # - gate_up_proj_bias: [E, 2 * expert_dim]
        # - down_proj: [E, expert_dim, hidden]
        # - down_proj_bias: [E, hidden]
        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            self.gate_up_proj = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, self.hidden_size, 2 * self.expert_dim),
                    dtype=dtype,
                    out_sharding=P("expert", None, "tensor"),
                )
            )
            self.gate_up_proj_bias = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, 2 * self.expert_dim),
                    dtype=dtype,
                    out_sharding=P("expert", "tensor"),
                )
            )
            self.down_proj = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, self.expert_dim, self.hidden_size),
                    dtype=dtype,
                    out_sharding=P("expert", "tensor", None),
                )
            )
            self.down_proj_bias = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, self.hidden_size),
                    dtype=dtype,
                    out_sharding=P("expert", None),
                )
            )

    def __call__(self, hidden_states: jax.Array, topk_weights: jax.Array, topk_ids: jax.Array):
        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            hidden_states_reshard = jax.sharding.reshard(hidden_states, P(None))
            topk_weights_reshard = jax.sharding.reshard(topk_weights, P(None))
            topk_ids_reshard = jax.sharding.reshard(topk_ids, P(None))

            result = shard_map(
                self._forward,
                mesh=self.moe_mesh,
                in_specs=(
                    P(None),
                    P(None),
                    P(None),
                    P("expert", None, "tensor"),
                    P("expert", "tensor"),
                    P("expert", "tensor", None),
                    P("expert", None),
                ),
                out_specs=P(None),
                check_vma=False,
            )(
                hidden_states_reshard,
                topk_weights_reshard,
                topk_ids_reshard,
                self.gate_up_proj.value,
                self.gate_up_proj_bias.value,
                self.down_proj.value,
                self.down_proj_bias.value,
            )

        output_pspec = P(*([None] * (result.ndim)))
        result = jax.sharding.reshard(result, NamedSharding(self.original_mesh, output_pspec))
        return result.astype(hidden_states.dtype)

    def _forward(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        gate_up_proj: jax.Array,
        gate_up_proj_bias: jax.Array,
        down_proj: jax.Array,
        down_proj_bias: jax.Array,
    ) -> jax.Array:
        # NOTE: ep_size==1 in our current TPU setup (no expert-parallel token dispatch).
        if hidden_states.ndim == 2:
            inputs_2d = hidden_states
            batch_size, seq_len = 1, inputs_2d.shape[0]
        else:
            batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            inputs_2d = jnp.reshape(hidden_states, (batch_size * seq_len, hidden_states.shape[-1]))

        flatten_selected_experts = jnp.ravel(topk_ids)
        sorted_selected_experts = jnp.argsort(flatten_selected_experts, stable=True)
        sorted_indices = sorted_selected_experts // self.top_k
        sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)
        expert_ids = jnp.take(flatten_selected_experts, indices=sorted_selected_experts, axis=0)

        group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts).astype(jnp.int32)
        expert_shard_id = jax.lax.axis_index("expert")
        group_offset = jnp.array(expert_shard_id * self.experts_per_device, dtype=jnp.int32)

        gate_up = gmm(
            lhs=sorted_inputs,
            rhs=gate_up_proj,
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            tiling=(
                min(512, sorted_inputs.shape[0]),
                min(1024, sorted_inputs.shape[1]),
                min(1024, gate_up_proj.shape[-1]),
            ),
            group_offset=group_offset,
        )
        gate_up = gate_up + gate_up_proj_bias[expert_ids]

        gate = gate_up[..., ::2]
        up = gate_up[..., 1::2]
        gate = jnp.minimum(gate, self.limit)
        up = jnp.clip(up, -self.limit, self.limit)

        glu = gate * jax.nn.sigmoid(gate * self.alpha)
        intermediate = (up + 1.0) * glu

        out = gmm(
            lhs=intermediate,
            rhs=down_proj,
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            tiling=(
                min(512, intermediate.shape[0]),
                min(1024, intermediate.shape[1]),
                min(1024, down_proj.shape[-1]),
            ),
            group_offset=group_offset,
        )
        if self.tp_size > 1:
            out = jax.lax.psum(out, "tensor")
        out = out + down_proj_bias[expert_ids]

        argsort_indices = jnp.argsort(sorted_selected_experts, stable=True)
        unsort_out = jnp.take(out, indices=argsort_indices, axis=0)

        total_tokens = topk_weights.shape[0]
        reshaped_out = unsort_out.reshape(total_tokens, self.top_k, -1).astype(jnp.float32)
        reshaped_w = topk_weights.astype(jnp.float32)
        combined = jnp.einsum("tke,tk->te", reshaped_out, reshaped_w).astype(self.dtype)

        if hidden_states.ndim == 2:
            return combined
        return combined.reshape(batch_size, seq_len, -1)


class GptOssAttention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
        layer_id: int,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.layer_id = int(layer_id)

        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        num_kv_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))
        rope_theta = float(getattr(config, "rope_theta", 10000.0))
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 4096))

        self.head_dim = head_dim
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * head_dim,
            use_bias=getattr(config, "attention_bias", False),
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * head_dim,
            use_bias=getattr(config, "attention_bias", False),
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * head_dim,
            use_bias=getattr(config, "attention_bias", False),
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            use_bias=getattr(config, "attention_bias", False),
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        sliding_window_size = 0
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and layer_types[layer_id] == "sliding_attention":
            sliding_window_size = int(getattr(config, "sliding_window", 0))

        self.rotary_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
            dtype=dtype,
        )
        self.sinks = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (num_heads,),
                dtype=dtype,
                out_sharding=jax.sharding.PartitionSpec(None),
            )
        )
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
        )

    def __call__(
        self,
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

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        out, _ = self.o_proj(attn_output)
        return out, kv_fused


class GptOssDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
        layer_id: int,
    ):
        self.layer_id = int(layer_id)
        self.self_attn = GptOssAttention(config, mesh=mesh, dtype=dtype, layer_id=layer_id)
        self.router = GptOssTopKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            mesh=mesh,
            dtype=dtype,
        )
        self.experts = GptOssExperts(config, mesh=mesh, dtype=dtype, layer_id=layer_id)

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        topk_weights, topk_ids = self.router(hidden_states)
        hidden_states = self.experts(hidden_states, topk_weights, topk_ids)

        return hidden_states, residual, kv_fused


class GptOssModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                GptOssDecoderLayer(config, mesh=mesh, dtype=dtype, layer_id=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.layers_to_capture: list[int] = []
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_callback_flag = []
        aux_hidden_states = []
        for layer in self.layers:
            if layer.layer_id in self.layers_to_capture:
                aux_hidden_states.append(hidden_states if residual is None else (hidden_states + residual))
            hidden_states, residual, kv_fused = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)
        return hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag


class GptOssForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = GptOssModel(config, mesh=mesh, dtype=dtype)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)
        self.capture_aux_hidden_states = False

    def set_dflash_layers_to_capture(self, layer_ids: list[int]):
        self.capture_aux_hidden_states = True
        # Match sglang-jax convention: for the ith layer, we capture output of (i-1)th.
        self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def get_embed_and_head(self):
        return (
            self.model.embed_tokens.embedding.value,
            self.lm_head.embedding.value,
        )

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )
        if not self.capture_aux_hidden_states:
            aux_hidden_states = None

        output = self.logits_processor(
            hidden_states, self.lm_head, logits_metadata, aux_hidden_states=aux_hidden_states
        )
        return output, layers_kv_fused, layers_callback_flag

    def load_weights(self, model_config: "ModelConfig"):
        loader = WeightLoader(model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype)
        loader.load_weights_from_safetensors(self._create_weight_mappings())
        logger.info("GPT-OSS weights loaded successfully!")

    def _create_weight_mappings(self) -> dict[str, WeightMapping]:
        mappings: dict[str, WeightMapping] = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            "lm_head.weight": WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
        }

        for i in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{i}"
            target_prefix = f"model.layers.{i}"
            layer_mappings = {
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
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.v_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.o_proj.weight",
                    sharding=("tensor", None),
                    transpose=True,
                ),
                f"{prefix}.self_attn.sinks": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.sinks",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.mlp.router.weight": WeightMapping(
                    target_path=f"{target_prefix}.router.weight",
                    sharding=(None, None),
                    transpose=True,
                ),
                f"{prefix}.mlp.router.bias": WeightMapping(
                    target_path=f"{target_prefix}.router.bias",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.mlp.experts.gate_up_proj": WeightMapping(
                    target_path=f"{target_prefix}.experts.gate_up_proj",
                    sharding=("expert", None, "tensor"),
                    transpose=False,
                ),
                f"{prefix}.mlp.experts.gate_up_proj_bias": WeightMapping(
                    target_path=f"{target_prefix}.experts.gate_up_proj_bias",
                    sharding=("expert", "tensor"),
                    transpose=False,
                ),
                f"{prefix}.mlp.experts.down_proj": WeightMapping(
                    target_path=f"{target_prefix}.experts.down_proj",
                    sharding=("expert", "tensor", None),
                    transpose=False,
                ),
                f"{prefix}.mlp.experts.down_proj_bias": WeightMapping(
                    target_path=f"{target_prefix}.experts.down_proj_bias",
                    sharding=("expert", None),
                    transpose=False,
                ),
            }

            if getattr(self.config, "attention_bias", False):
                layer_mappings.update(
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
                            kv_head_padding=True,
                        ),
                        f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                            target_path=f"{target_prefix}.self_attn.v_proj.bias",
                            sharding=("tensor",),
                            transpose=False,
                            kv_head_padding=True,
                        ),
                        f"{prefix}.self_attn.o_proj.bias": WeightMapping(
                            target_path=f"{target_prefix}.self_attn.o_proj.bias",
                            sharding=(None,),
                            transpose=False,
                        ),
                    }
                )

            mappings.update(layer_mappings)

        return mappings


EntryClass = GptOssForCausalLM
