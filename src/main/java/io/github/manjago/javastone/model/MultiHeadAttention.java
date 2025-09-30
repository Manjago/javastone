package io.github.manjago.javastone.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;

public class MultiHeadAttention {

    private final int numHeads;
    private final int headDim;
    private final LinearLayer q_proj;
    private final LinearLayer k_proj;
    private final LinearLayer v_proj;
    private final LinearLayer o_proj;

    public MultiHeadAttention(int embedDim, int numHeads,
                              LinearLayer q_proj, LinearLayer k_proj,
                              LinearLayer v_proj, LinearLayer o_proj) {
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.q_proj = q_proj;
        this.k_proj = k_proj;
        this.v_proj = v_proj;
        this.o_proj = o_proj;
    }

    public NDArray forward(NDArray input) {
        Shape inputShape = input.getShape();
        long batchSize = inputShape.get(0);
        long seqLen = inputShape.get(1);

        NDArray q = q_proj.forward(input);
        NDArray k = k_proj.forward(input);
        NDArray v = v_proj.forward(input);

        q = q.reshape(batchSize, seqLen, numHeads, headDim).transpose(0, 2, 1, 3);
        k = k.reshape(batchSize, seqLen, numHeads, headDim).transpose(0, 2, 1, 3);
        v = v.reshape(batchSize, seqLen, numHeads, headDim).transpose(0, 2, 1, 3);

        NDArray scores = q.matMul(k.transpose(0, 1, 3, 2));

        float scale = (float) (1.0 / Math.sqrt(headDim));
        NDArray attentionWeights = scores.mul(scale).softmax(-1);
        NDArray attentionOutput = attentionWeights.matMul(v);

        NDArray concatenatedOutput = attentionOutput.transpose(0, 2, 1, 3).reshape(batchSize, seqLen, -1);

        return o_proj.forward(concatenatedOutput);
    }
    
    // main метод для теста можно пока убрать, он нам больше не нужен
}