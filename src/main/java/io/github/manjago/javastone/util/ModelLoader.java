package io.github.manjago.javastone.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ModelLoader {

    public static void main(String[] args) throws IOException {
        // Укажи путь к папке с твоей моделью
        Path modelPath = Paths.get(System.getProperty("user.home"), "models", "gemma-2b");
        System.out.println("Loading model from: " + modelPath);
        
        // --- 1. "Пощупаем" загрузку конфига ---
        System.out.println("\n--- Loading Config ---");
        
        ObjectMapper mapper = new ObjectMapper();
        GemmaConfig config = mapper.readValue(modelPath.resolve("config.json").toFile(), GemmaConfig.class);

        System.out.println("Config loaded successfully!");
        System.out.println("Hidden Size: " + config.hiddenSize());
        System.out.println("Num Layers: " + config.numHiddenLayers());
        System.out.println("Num Heads: " + config.numAttentionHeads());
    }
}