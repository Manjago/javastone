package io.github.manjago.javastone.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Реализация линейного (полносвязного) слоя нейронной сети.
 * Выполняет операцию: output = input * weight^T + bias
 */
public class LinearLayer {

    private final NDArray weight;
    private final NDArray bias;

    /**
     * Конструктор слоя.
     * @param weight Матрица весов. Shape: (output_features, input_features)
     * @param bias Вектор смещения. Shape: (output_features). Может быть null.
     */
    public LinearLayer(NDArray weight, NDArray bias) {
        this.weight = weight;
        this.bias = bias;
    }

    /**
     * Прямой проход через слой.
     * @param input Входной тензор. Shape: (batch_size, input_features)
     * @return Выходной тензор. Shape: (batch_size, output_features)
     */
    public NDArray forward(NDArray input) {
        // Выполняем матричное умножение: input.matMul(weight.transpose())
        NDArray output = input.matMul(weight.transpose());
        if (bias != null) {
            // И добавляем смещение, если оно есть
            output = output.add(bias);
        }
        return output;
    }

    public Shape getWeightShape() {
        return weight.getShape();
    }

    // --- Тестовый блок для проверки ---
    public static void main(String[] args) {
        System.out.println("--- Testing LinearLayer ---");

        try (NDManager manager = NDManager.newBaseManager()) {
            // Определим параметры: 10 входных фичей, 5 выходных
            long inputFeatures = 10;
            long outputFeatures = 5;

            // Создадим фейковые веса и смещение
            // Обрати внимание на форму весов: (output, input)
            NDArray fakeWeights = manager.randomNormal(new Shape(outputFeatures, inputFeatures));
            NDArray fakeBias = manager.randomNormal(new Shape(outputFeatures));

            // Создаем наш слой
            LinearLayer layer = new LinearLayer(fakeWeights, fakeBias);
            System.out.println("Layer created. Weight shape: " + layer.getWeightShape());

            // Создадим фейковый батч из 2-х векторов
            long batchSize = 2;
            NDArray input = manager.randomNormal(new Shape(batchSize, inputFeatures));
            System.out.println("Input batch shape: " + input.getShape());

            // Прогоняем данные через слой
            NDArray result = layer.forward(input);

            System.out.println("Result shape: " + result.getShape()); // Ожидаем (2, 5)
            System.out.println("Result data:\n" + result);
            System.out.println("Test successful!");
        }
    }
}