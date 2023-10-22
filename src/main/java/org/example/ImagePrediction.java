package org.example;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImagePrediction {
    public static void main(String[] args) {
        String modelPath = "src/main/modelTrained/model.zip"; // Ruta al archivo ZIP del modelo

        // Cargar el modelo previamente entrenado
        MultiLayerNetwork model;
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(modelPath);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        // Cargar una imagen de 28x28 píxeles (que sea en escala de grises)
        BufferedImage image = loadImage("src\\main\\resources\\test9modified.jpg");

        // Realizar la predicción
        INDArray imageArray = preprocessImage(image);
        INDArray output = model.output(imageArray);

        // Mostrar el resultado
        System.out.println("Predicción de clase: " + output.argMax(1));
    }

    private static BufferedImage loadImage(String imagePath) {
        try {
            return ImageIO.read(new File(imagePath));
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private static INDArray preprocessImage(BufferedImage image) {
        // Asegúrate de que la imagen tenga el tamaño correcto (28x28 píxeles)
        if (image.getWidth() != 28 || image.getHeight() != 28) {
            throw new IllegalArgumentException("La imagen debe tener un tamaño de 28x28 píxeles.");
        }

        int width = image.getWidth();
        int height = image.getHeight();
        float[] pixelData = new float[784]; // Vector de 784 valores

        int pixelIdx = 0; // Índice del pixel en el vector

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int grayValue = image.getRGB(x, y) & 0xFF; // Obtener el valor en escala de grises
                pixelData[pixelIdx++] = (float) grayValue / 255.0f;
            }
        }

        return org.nd4j.linalg.factory.Nd4j.create(pixelData);
    }

}
