package io.github.manjago.javastone;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

public class JavastoneApp {

    public static void main(String[] args) {
        System.out.println("Project Javastone is starting!");
        System.out.println("--- Step 1: Hello, Tensor! ---");

        // NDManager - это фабрика для создания тензоров.
        // Он управляет памятью. Использование try-with-resources обязательно!
        // Это гарантирует, что вся память, выделенная под тензоры, будет освобождена.
        try (NDManager manager = NDManager.newBaseManager()) {

            // 1. Создаем наш первый тензор (вектор) из простого Java-массива
            final float[] vectorData = {1.0f, 2.0f, 3.0f, 4.0f};
            final NDArray vector = manager.create(vectorData);

            System.out.println("Создали вектор:");
            System.out.println("Данные: " + vector);
            System.out.println("Форма (shape): " + vector.getShape()); // Shape - это размерность тензора
            System.out.println("------------------------------------");


            // 2. Создадим матрицу 2x3
            final float[][] matrixData = {{1, 2, 3}, {4, 5, 6}};
            final NDArray matrixA = manager.create(matrixData);

            System.out.println("Создали матрицу A:");
            System.out.println("Данные:\n" + matrixA);
            System.out.println("Форма (shape): " + matrixA.getShape());
            System.out.println("------------------------------------");


            // 3. Выполним базовые операции
            // Прибавим число 10 к каждому элементу матрицы
            final NDArray matrixAPlus10 = matrixA.add(10);
            System.out.println("Матрица A + 10:\n" + matrixAPlus10);
            System.out.println("------------------------------------");

            // 4. Главная операция в нейросетях - матричное умножение!
            // Умножим нашу матрицу A (2x3) на другую матрицу B (3x2)
            // Результат должен получиться (2x2)
            float[][] matrixBData = {{7, 8}, {9, 10}, {11, 12}};
            NDArray matrixB = manager.create(matrixBData);

            System.out.println("Создали матрицу B:");
            System.out.println("Данные:\n" + matrixB);
            System.out.println("Форма (shape): " + matrixB.getShape());
            System.out.println("------------------------------------");

            NDArray resultMatrix = matrixA.matMul(matrixB);

            System.out.println("Результат умножения A * B:");
            System.out.println("Данные:\n" + resultMatrix);
            System.out.println("Форма (shape): " + resultMatrix.getShape());
            System.out.println("------------------------------------");

        } // Вся память автоматически очистится здесь
    }
}