#include "models.h"

llm_build_glm4_moe_lite::llm_build_glm4_moe_lite(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    // GLM4_MOE_LITE uses Multi-head Latent Attention (MLA) like DeepSeek2
    // Combined with MoE + shared experts like GLM4_MOE

    const bool is_mla = (hparams.n_embd_head_k_mla != 0 && hparams.n_embd_head_v_mla != 0);
    GGML_ASSERT(is_mla && "GLM4_MOE_LITE requires MLA parameters");

    // MLA dimensions after decompression
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla;  // qk_nope_head_dim + qk_rope_head_dim

    const int64_t n_embd_head_qk_rope = hparams.n_rot;        // qk_rope_head_dim
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;
    const uint32_t q_lora_rank = hparams.n_lora_q;

    // Scaling factor with YaRN mscale support (same as DeepSeek2)
    // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
    float kq_scale;
    if (hparams.rope_yarn_log_mul != 0.0f && ext_factor >= 0.0f && freq_scale != 1.0f) {
        // YaRN is enabled - apply mscale^2 adjustment
        const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
        const float mscale = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
        kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));
    } else {
        // No YaRN - simple scaling
        kq_scale = 1.0f / sqrtf(float(n_embd_head_k));
    }

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Skip final NextN layer(s) - they're for multi-token prediction, not main forward pass
    const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;

    for (int il = 0; il < n_transformer_layers; ++il) {
        ggml_tensor * inpSA = inpL;

        // Pre-attention norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // MLA self-attention
        {
            ggml_tensor * q = NULL;

            // Q projection with LoRA compression: Q_A -> norm -> Q_B
            if (q_lora_rank > 0) {
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q_a", il);

                q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q, "q_a_norm", il);

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
            }

            // Split Q into nope (non-position-encoded) and pe (position-encoded) parts
            // {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                             ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            // {n_embd_head_qk_rope, n_head, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head,
                ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            // KV compression: projects to {kv_lora_rank + n_embd_head_qk_rope, n_tokens}
            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // Split into compressed KV {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            // and K rope part {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            // Apply RoPE to position-encoded parts
            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe_rope", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe_rope", il);

            // Normalize compressed KV
            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr_norm", il);

            // MLA with absorption optimization (converts to MQA)
            // Absorb K_B into Q_nope: Q_nope @ K_B^T
            // {n_embd_head_qk_nope, n_tokens, n_head}
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            cb(q_nope, "q_nope_perm", il);

            // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
            ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
            cb(q_nope_absorbed, "q_nope_absorbed", il);

            // {kv_lora_rank, n_head, n_tokens}
            q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
            cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

            // Concat rope + absorbed parts for Q: {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
            ggml_tensor * Qcur = ggml_concat(ctx0, q_pe, q_nope_absorbed, 0);
            cb(Qcur, "Qcur", il);

            kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
            cb(kv_cmpr, "kv_cmpr_reshape", il);

            // K: {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
            ggml_tensor * Kcur = ggml_concat(ctx0, k_pe, kv_cmpr, 0);
            cb(Kcur, "Kcur", il);

            // V: {kv_lora_rank, 1, n_tokens}
            ggml_tensor * Vcur = kv_cmpr;
            cb(Vcur, "Vcur", il);

            // MLA with absorption converts to MQA (1 KV head)
            // V_B is applied after attention via the last parameter
            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
        }

        if (il == n_transformer_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // Post-attention norm (GLM4 style)
        cur = build_norm(ffn_inp, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "post_attn_norm", il);

        // FFN: dense for first n_layer_dense_lead layers, then MoE
        if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
            // Dense FFN layer
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE layer
            ggml_tensor * routed_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(routed_out, "ffn_moe_out", il);

            // Shared expert (if present)
            if (hparams.n_expert_shared > 0) {
                ggml_tensor * shared_out = build_ffn(cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(shared_out, "ffn_shexp_out", il);

                // Sum routed + shared
                cur = ggml_add(ctx0, routed_out, shared_out);
            } else {
                cur = routed_out;
            }
            cb(cur, "ffn_out", il);
        }

        // FFN residual
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
