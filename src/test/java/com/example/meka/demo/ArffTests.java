package com.example.meka.demo;

import org.junit.Test;

import meka.classifiers.MultiXClassifier;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLUtils;
import meka.core.Result;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class ArffTests {

    private static final String NO = "no";
    private static final String YES = "yes";
    private static final String WARM = "warm";
    private static final String COLD = "cold";
    private static final String HOT = "hot";
    private static final String CLEAR = "clear";
    private static final String RAIN = "rain";
    private static final String SNOW = "snow";
    private static final String LAZY = "lazy";
    private static final String ENERGETIC = "energetic";

    private static final double THRESHOLD = 0.7; // 70% confident

    private MultiXClassifier classifier;

    @Test
    public void testArff() throws Exception {

        int howMany = 1;

        List<String> trainingArffLines = buildArffFormat();
        List<String> predictionArffLines = buildArffFormat();

        createTrainingInstance(YES, NO, NO, YES, WARM, CLEAR, LAZY, trainingArffLines, howMany);
        createTrainingInstance(YES, YES, NO, NO, COLD, SNOW, ENERGETIC, trainingArffLines, howMany);
        createTrainingInstance(YES, NO, NO, YES, HOT, RAIN, LAZY, trainingArffLines, howMany);

        createPredictionInstance(null, null, LAZY, predictionArffLines, howMany);
        createPredictionInstance(COLD, SNOW, LAZY, predictionArffLines, howMany);

        printLines(trainingArffLines);
        printLines(predictionArffLines);

        Instances trainInstances = getInstances(trainingArffLines);
        Instances predictionInstances = getInstances(predictionArffLines);

        Result result = getPrediction(trainInstances, predictionInstances);

        assertPredictions(result, 0, true, false, false, true);
    }

    private Result getPrediction(Instances trainInstances, Instances predictionInstances) throws Exception {
        classifier = new CC();

        Result result = Evaluation.evaluateModel(classifier, trainInstances, predictionInstances);

        System.out.println(Result.getResultAsString(result));
        // System.out.println(result.toString());

        return result;
    }

    private void createPredictionInstance(String temp, String cond, String enrgy, List<String> arffLines,
            int howMany) {

        createTrainingInstance(null, null, null, null, temp, cond, enrgy, arffLines, howMany);
    }

    private void createTrainingInstance(String fb, String ski, String book, String park, String temp, String cond,
            String enrgy, List<String> arffLines, int howMany) {

        StringBuilder instance = new StringBuilder();

        for (int x = 0; x < howMany; x++) {
            if (fb == null)
                instance.append("?");
            else
                instance.append(fb);
            instance.append(",");

            if (ski == null)
                instance.append("?");
            else
                instance.append(ski);
            instance.append(",");

            if (book == null)
                instance.append("?");
            else
                instance.append(book);
            instance.append(",");

            if (park == null)
                instance.append("?");
            else
                instance.append(park);
            instance.append(",");

            if (temp == null)
                instance.append("?");
            else
                instance.append(temp);
            instance.append(",");

            if (cond == null)
                instance.append("?");
            else
                instance.append(cond);
            instance.append(",");

            if (enrgy == null)
                instance.append("?");
            else
                instance.append(enrgy);

            arffLines.add(instance.toString());
        }
    }

    private Instances getInstances(List<String> list) throws Exception {

        InputStream stream = new ByteArrayInputStream(stringifyList(list).getBytes());
        Instances instance = DataSource.read(stream);
        MLUtils.prepareData(instance);

        return instance;

    }

    private List<String> buildArffFormat() {

        List<String> lines = new ArrayList<>();

        lines.add("@relation 'WeatherConditions: -C 4'");
        lines.add("@attribute football {" + NO + "," + YES + "}");
        lines.add("@attribute goskiing {" + NO + "," + YES + "}");
        lines.add("@attribute gotopark {" + NO + "," + YES + "}");
        lines.add("@attribute readabook {" + NO + "," + YES + "}");
        lines.add("@attribute temperature {" + WARM + "," + COLD + "," + HOT + "}");
        lines.add("@attribute conditions {" + CLEAR + "," + RAIN + "," + SNOW + "}");
        lines.add("@attribute energetic {" + LAZY + "," + ENERGETIC + "}");

        lines.add("@data");

        return lines;
    }

    private String stringifyList(List<String> list) {

        StringBuilder builder = new StringBuilder();
        for (String str : list) {
            builder.append(str);
            builder.append("\n");
        }

        return builder.toString();
    }

    private void assertPredictions(Result result, int index, boolean football, boolean ski, boolean book,
            boolean park) {

        int[][] predictions = result.allPredictions(THRESHOLD);
        double[][] percentage = result.allPredictions();
        printPercentages(percentage);

        assertEquals("football result not expected", football ? 1 : 0, predictions[index][0]);

        // should have other assertions here, too

    }

    private void printLines(List<String> arffLines) {

        for (String line : arffLines) {
            System.out.println(line);
        }
    }

    private void printPercentages(double[][] percentage) {

        for (double[] outer : percentage) {
            for (double inner : outer) {
                System.out.print("[" + inner + "]");
            }
            System.out.println("");
        }
    }
}
