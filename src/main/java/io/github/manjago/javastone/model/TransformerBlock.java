package io.github.manjago.javastone.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;

public class TransformerBlock {

    private final MultiHeadAttention attention;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final LinearLayer ffn1;
    private final LinearLayer ffn2;

    public TransformerBlock(MultiHeadAttention attention, LayerNorm norm1, LayerNorm norm2,
                            LinearLayer ffn1, LinearLayer ffn2) {
        this.attention = attention;
        this.norm1 = norm1;
        this.norm2 = norm2;
        this.ffn1 = ffn1;
        this.ffn2 = ffn2;
    }

    public NDArray forward(NDArray input) {
        NDArray normInput = norm1.forward(input);
        NDArray attentionOutput = attention.forward(normInput);
        NDArray residual1 = input.add(attentionOutput);

        NDArray normResidual1 = norm2.forward(residual1);
        NDArray ffnOutput = ffn1.forward(normResidual1);
        ffnOutput = Activation.gelu(ffnOutput);
        ffnOutput = ffn2.forward(ffnOutput);
        NDArray finalOutput = residual1.add(ffnOutput);

        return finalOutput;
    }

    public static void main(String[] args) {
        // ... (main метод остается здесь для теста)
        System.out.println("--- Testing TransformerBlock ---");

        try (NDManager manager = NDManager.newBaseManager()) {
            int embedDim = 12;
            int numHeads = 4;
            int ffnHiddenDim = 48;
            int batchSize = 1;
            int seqLen = 5;

            LinearLayer q_proj = new LinearLayer(manager.randomNormal(new Shape(embedDim, embedDim)), null);
            LinearLayer k_proj = new LinearLayer(manager.randomNormal(new Shape(embedDim, embedDim)), null);
            LinearLayer v_proj = new LinearLayer(manager.randomNormal(new Shape(embedDim, embedDim)), null);
            LinearLayer o_proj = new LinearLayer(manager.randomNormal(new Shape(embedDim, embedDim)), null);
            MultiHeadAttention attention = new MultiHeadAttention(embedDim, numHeads, q_proj, k_proj, v_proj, o_proj);

            LayerNorm norm1 = new LayerNorm(manager.ones(new Shape(embedDim)), manager.zeros(new Shape(embedDim)), 1e-5f);
            LayerNorm norm2 = new LayerNorm(manager.ones(new Shape(embedDim)), manager.zeros(new Shape(embedDim)), 1e-5f);
            
            LinearLayer ffn1 = new LinearLayer(manager.randomNormal(new Shape(ffnHiddenDim, embedDim)), null);
            LinearLayer ffn2 = new LinearLayer(manager.randomNormal(new Shape(embedDim, ffnHiddenDim)), null);

            TransformerBlock block = new TransformerBlock(attention, norm1, norm2, ffn1, ffn2);

            NDArray input = manager.randomNormal(new Shape(batchSize, seqLen, embedDim));
            System.out.println("Input shape: " + input.getShape());

            NDArray result = block.forward(input);
            System.out.println("Result shape: " + result.getShape());
            
            if (result.getShape().equals(input.getShape())) {
                System.out.println("\nShape test successful! Our composite block preserves the shape.");
            } else {
                System.err.println("\nShape test FAILED!");
            }
        }
    }
}