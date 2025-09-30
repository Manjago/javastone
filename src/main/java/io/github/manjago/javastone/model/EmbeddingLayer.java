package io.github.manjago.javastone.model;

import ai.djl.ndarray.NDArray;

/**
 * Простая реализация слоя эмбеддингов.
 * По сути, это просто таблица поиска (lookup table).
 */
public class EmbeddingLayer {

    private final NDArray weight;

    /**
     * @param weight Матрица весов эмбеддингов. Shape: (vocab_size, embed_dim)
     */
    public EmbeddingLayer(NDArray weight) {
        this.weight = weight;
    }

    /**
     * Превращает тензор с индексами токенов в тензор векторов.
     * @param input Тензор с индексами. Shape: (batch_size, seq_len)
     * @return Тензор с эмбеддингами. Shape: (batch_size, seq_len, embed_dim)
     */
    public NDArray forward(NDArray input) {
        // DJL позволяет "выбирать" строки из тензора по индексам с помощью get()
        return weight.get(input);
    }
}