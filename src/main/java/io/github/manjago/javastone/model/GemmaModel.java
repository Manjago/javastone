package io.github.manjago.javastone.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.util.List;

/**
 * Полная архитектура модели Gemma.
 */
public class GemmaModel {

    private final EmbeddingLayer embedding;
    private final List<TransformerBlock> blocks;
    private final LayerNorm finalNorm;
    private final LinearLayer outputLayer; // Также называется "LM Head"

    public GemmaModel(EmbeddingLayer embedding, List<TransformerBlock> blocks, LayerNorm finalNorm, LinearLayer outputLayer) {
        this.embedding = embedding;
        this.blocks = blocks;
        this.finalNorm = finalNorm;
        this.outputLayer = outputLayer;
    }

    public NDArray forward(NDArray tokens) {
        // 1. Превращаем токены в векторы
        NDArray x = embedding.forward(tokens);

        // 2. Последовательно прогоняем через все блоки
        for (TransformerBlock block : blocks) {
            x = block.forward(x);
        }

        // 3. Финальная нормализация
        x = finalNorm.forward(x);

        // 4. Проецируем на словарь для получения логитов
        return outputLayer.forward(x);
    }

    // --- Тестовый блок для проверки всей архитектуры ---
    public static void main(String[] args) {
        System.out.println("--- Testing Full GemmaModel Architecture ---");
        try (NDManager manager = NDManager.newBaseManager()) {
            // --- Зададим параметры для "игрушечной" модели ---
            int vocabSize = 100;    // Размер словаря
            int embedDim = 12;      // Размер эмбеддинга
            int numHeads = 4;       // Количество голов
            int ffnHiddenDim = 48;  // Скрытый слой в FFN
            int numBlocks = 2;      // У нас будет всего 2 блока вместо 18

            // --- Создаем фейковые веса и слои ---
            // 1. Embedding
            EmbeddingLayer embedding = new EmbeddingLayer(manager.randomNormal(new Shape(vocabSize, embedDim)));

            // 2. Transformer Blocks (создадим 2 штуки)
            TransformerBlock block1 = TestUtils.createFakeTransformerBlock(manager, embedDim, numHeads, ffnHiddenDim);
            TransformerBlock block2 = TestUtils.createFakeTransformerBlock(manager, embedDim, numHeads, ffnHiddenDim);
            List<TransformerBlock> blocks = List.of(block1, block2);

            // 3. Final Layers
            LayerNorm finalNorm = new LayerNorm(manager.ones(new Shape(embedDim)), manager.zeros(new Shape(embedDim)), 1e-5f);
            LinearLayer outputLayer = new LinearLayer(manager.randomNormal(new Shape(vocabSize, embedDim)), null);
            
            // --- Собираем модель ---
            GemmaModel model = new GemmaModel(embedding, blocks, finalNorm, outputLayer);
            System.out.println("Model assembled successfully with " + numBlocks + " blocks.");

            // --- Создаем фейковый вход ---
            int batchSize = 1;
            int seqLen = 5;
            // Вход - это просто 5 чисел (индексов токенов)
            NDArray inputTokens = manager.create(new long[]{10, 25, 5, 78, 99}, new Shape(batchSize, seqLen));
            System.out.println("\nInput token IDs shape: " + inputTokens.getShape());
            
            // --- Выполняем полный прямой проход! ---
            NDArray logits = model.forward(inputTokens);
            System.out.println("Final logits shape: " + logits.getShape());
            
            // --- Главная проверка ---
            Shape expectedShape = new Shape(batchSize, seqLen, vocabSize);
            if (logits.getShape().equals(expectedShape)) {
                System.out.println("\nShape test successful! The model produces logits with the correct shape.");
                System.out.println("Expected shape: " + expectedShape);
            } else {
                System.err.println("\nShape test FAILED!");
            }
        }
    }
}

/**
 * Вспомогательный класс, чтобы не загромождать main метод.
 */
class TestUtils {
    public static TransformerBlock createFakeTransformerBlock(NDManager manager, int embedDim, int numHeads, int ffnHiddenDim) {
        LinearLayer q_proj = new LinearLayer(manager.randomNormal(new Shape(embedDim, embedDim)), null);
        LinearLayer k_proj = new LinearLayer(manager.randomNormal(new Shape(embedDim, embedDim)), null);
        LinearLayer v_proj = new LinearLayer(manager.randomNormal(new Shape(embedDim, embedDim)), null);
        LinearLayer o_proj = new LinearLayer(manager.randomNormal(new Shape(embedDim, embedDim)), null);
        MultiHeadAttention attention = new MultiHeadAttention(embedDim, numHeads, q_proj, k_proj, v_proj, o_proj);
        LayerNorm norm1 = new LayerNorm(manager.ones(new Shape(embedDim)), manager.zeros(new Shape(embedDim)), 1e-5f);
        LayerNorm norm2 = new LayerNorm(manager.ones(new Shape(embedDim)), manager.zeros(new Shape(embedDim)), 1e-5f);
        LinearLayer ffn1 = new LinearLayer(manager.randomNormal(new Shape(ffnHiddenDim, embedDim)), null);
        LinearLayer ffn2 = new LinearLayer(manager.randomNormal(new Shape(embedDim, ffnHiddenDim)), null);
        return new TransformerBlock(attention, norm1, norm2, ffn1, ffn2);
    }
}