"""Write a tiny random-weight llama-architecture GGUF, so the local llama.cpp path can be tested end to end
without downloading anything. Its prose is noise, but grammar-constrained output must still parse."""

from __future__ import annotations

from pathlib import Path


DEFAULT_TEMPLATE = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant: "

# Gemma-2's published template: rejects a system message and demands strict user/assistant alternation.
GEMMA2_TEMPLATE = (
    "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}"
    "{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
    "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}"
    "{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
)

# A Mistral-v0.3-style template: an optional leading system message, then strict alternation.
STRICT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
    "{% set sys = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set sys = '' %}{% endif %}"
    "{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('After the optional system message, conversation roles must alternate "
    "user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + sys + ' ' + message['content'] + ' [/INST]' }}{% else %}{{ message['content'] }}{% endif %}"
    "{% endfor %}"
)


def write_tiny_llama(path: Path, seed: int = 0, chat_template: str = DEFAULT_TEMPLATE) -> Path:
    import gguf
    import numpy as np

    rng = np.random.default_rng(seed)
    n_embd, n_head, n_ff, n_layer = 32, 4, 64, 1
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
    types = [gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL] + [gguf.TokenType.BYTE] * 256

    w = gguf.GGUFWriter(str(path), "llama")
    w.add_name("voidlit-test-tiny")
    w.add_context_length(4096)
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head)
    w.add_rope_dimension_count(n_embd // n_head)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * len(tokens))
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_chat_template(chat_template)

    def rand(*shape: int) -> None:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    ones = np.ones(n_embd, dtype=np.float32)
    w.add_tensor("token_embd.weight", rand(len(tokens), n_embd))
    w.add_tensor("output_norm.weight", ones)
    w.add_tensor("output.weight", rand(len(tokens), n_embd))
    for i in range(n_layer):
        w.add_tensor(f"blk.{i}.attn_norm.weight", ones)
        for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
            w.add_tensor(f"blk.{i}.{name}.weight", rand(n_embd, n_embd))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", ones)
        w.add_tensor(f"blk.{i}.ffn_gate.weight", rand(n_ff, n_embd))
        w.add_tensor(f"blk.{i}.ffn_up.weight", rand(n_ff, n_embd))
        w.add_tensor(f"blk.{i}.ffn_down.weight", rand(n_embd, n_ff))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path
