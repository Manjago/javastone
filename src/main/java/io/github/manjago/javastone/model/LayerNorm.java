package io.github.manjago.javastone.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Реализация Layer Normalization.
 * Стабилизирует выходы слоев, приводя их к стандартному распределению.
 */
public class LayerNorm {

    private final NDArray gamma; // learnable weight (scale)
    private final NDArray beta;  // learnable bias (shift)
    private final float epsilon;

    /**
     * @param gamma Вектор масштабирования. Shape: (features)
     * @param beta Вектор сдвига. Shape: (features)
     * @param epsilon Малое число для предотвращения деления на ноль.
     */
    public LayerNorm(NDArray gamma, NDArray beta, float epsilon) {
        this.gamma = gamma;
        this.beta = beta;
        this.epsilon = epsilon;
    }

    public NDArray forward(NDArray input) {
        // 1. & 2. Рассчитываем среднее и дисперсию вдоль последней оси (оси фичей)
        // new int[]{-1} означает "последняя ось"
        // true (keepDims) сохраняет размерность для корректного вычитания (broadcasting)
        NDArray mean = input.mean(new int[]{-1}, true);
        NDArray variance = input.sub(mean).pow(2).mean(new int[]{-1}, true);

        // 3. Нормализуем
        NDArray normalized = input.sub(mean).div(variance.add(epsilon).sqrt());

        // 4. Масштабируем и сдвигаем
        return normalized.mul(gamma).add(beta);
    }

    // --- Тестовый блок для проверки ---
    public static void main(String[] args) {
        System.out.println("--- Testing LayerNorm ---");

        try (NDManager manager = NDManager.newBaseManager()) {
            long features = 6;
            long batchSize = 2;

            // В реальной модели gamma и beta загружаются из весов.
            // Для теста инициализируем их стандартными значениями:
            // gamma = 1 (ничего не масштабируем), beta = 0 (ничего не сдвигаем).
            NDArray gamma = manager.ones(new Shape(features));
            NDArray beta = manager.zeros(new Shape(features));

            LayerNorm layerNorm = new LayerNorm(gamma, beta, 1e-5f);

            // Создадим фейковый входной тензор с разным масштабом
            NDArray input = manager.create(new float[][]{
                {1, 2, 3, 4, 5, 6},
                {-100, -50, 0, 50, 100, 150}
            });
            System.out.println("Input data:\n" + input);

            NDArray result = layerNorm.forward(input);
            System.out.println("\nNormalized result:\n" + result);

            // "Пощупаем" результат: проверим, что среднее близко к 0, а ст. отклонение к 1.
            // Для этого рассчитаем их для нашего результата.
            NDArray resultMean = result.mean(new int[]{-1});
            NDArray resultStd = result.sub(result.mean(new int[]{-1}, true)).pow(2).mean(new int[]{-1}, true).sqrt();

            System.out.println("\nVerification:");
            System.out.println("Mean of the result (should be ~0): " + resultMean);
            System.out.println("Std dev of the result (should be ~1): " + resultStd);
            System.out.println("\nTest successful!");
        }
    }
}