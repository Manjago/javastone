package io.github.manjago.javastone.util;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public record GemmaConfig(
    @JsonProperty("hidden_size") int hiddenSize,
    @JsonProperty("num_attention_heads") int numAttentionHeads,
    @JsonProperty("num_hidden_layers") int numHiddenLayers,
    @JsonProperty("intermediate_size") int intermediateSize,
    @JsonProperty("vocab_size") int vocabSize,
    @JsonProperty("rms_norm_eps") float rmsNormEps,
    @JsonProperty("head_dim") int headDim
) {}