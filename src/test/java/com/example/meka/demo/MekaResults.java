package com.example.meka.demo;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

import org.junit.Before;
import org.junit.Test;

import meka.classifiers.MultiXClassifier;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLUtils;
import meka.core.Result;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MekaResults {

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
    private static final String SKIING = "skiing";
    private static final String SNOWBOARDING = "snowboarding";
    private static final String SWINGSET = "swingset";
    private static final String MERRYGOROUND = "merrygoround";
    private static final String WALK = "walk";
    private static final String MALE = "male";
    private static final String FEMALE = "female";

    private Attribute watchFootball;
    private Attribute goSkiing;
    private Attribute goToPark;
    private Attribute readABook;
    private Attribute temperature;
    private Attribute conditions;
    private Attribute gender;
    private Attribute energy;

    private ArrayList<Attribute> attributes;

    private MultiXClassifier classifier;

    @Before
    public void setup() {

        List<String> yesOrNoList = new ArrayList<>();
        yesOrNoList.add(NO);
        yesOrNoList.add(YES);

        List<String> skiList = new ArrayList<>();
        skiList.add(NO);
        skiList.add(SKIING);
        skiList.add(SNOWBOARDING);

        List<String> parkList = new ArrayList<>();
        parkList.add(NO);
        parkList.add(SWINGSET);
        parkList.add(MERRYGOROUND);
        parkList.add(WALK);

        List<String> temperatureList = new ArrayList<>();
        temperatureList.add(WARM);
        temperatureList.add(COLD);
        temperatureList.add(HOT);

        List<String> weatherConditionsList = new ArrayList<>();
        weatherConditionsList.add(CLEAR);
        weatherConditionsList.add(RAIN);
        weatherConditionsList.add(SNOW);

        List<String> energyLevelList = new ArrayList<>();
        energyLevelList.add(LAZY);
        energyLevelList.add(ENERGETIC);

        List<String> genderList = new ArrayList<>();
        genderList.add(MALE);
        genderList.add(FEMALE);

        // these are to be predicted based on conditions
        watchFootball = new Attribute("football", yesOrNoList);
        goSkiing = new Attribute("goskiing", skiList);
        goToPark = new Attribute("gotopark", parkList);
        readABook = new Attribute("readabook", yesOrNoList);

        // these are conditions, will predict the above activities
        temperature = new Attribute("temperature", temperatureList);
        conditions = new Attribute("conditions", weatherConditionsList);
        gender = new Attribute("gender", genderList);
        energy = new Attribute("energetic", energyLevelList);

        attributes = new ArrayList<>();
        attributes.add(watchFootball);
        attributes.add(goSkiing);
        attributes.add(goToPark);
        attributes.add(readABook);
        attributes.add(temperature);
        attributes.add(conditions);
        attributes.add(gender);
        attributes.add(energy);

    }

    @Test
    public void simpleTests() throws Exception {

        Instances trainInstances = getInstances();
        Instances predictInstances = new Instances("WhatToDo", attributes, 1);

        trainInstances.setClassIndex(4);
        predictInstances.setClassIndex(4);

        createPredictInstance(COLD, SNOW, ENERGETIC, MALE, predictInstances);
        createPredictInstance(WARM, CLEAR, LAZY, FEMALE, predictInstances);
        createPredictInstance(COLD, SNOW, LAZY, FEMALE, predictInstances);
        createPredictInstance(COLD, SNOW, ENERGETIC, FEMALE, predictInstances);
        createPredictInstance(COLD, RAIN, ENERGETIC, MALE, predictInstances);

        classifier = new CC();

        Result result = Evaluation.evaluateModel(classifier, trainInstances, predictInstances);

        displayResults(result, predictInstances);

        System.out.println("\n\nFiltered data... \n");

        result = Evaluation.evaluateModel(classifier, filterSomeData(trainInstances), predictInstances);

        displayResults(result, predictInstances);

    }

    private void displayResults(Result result, Instances predictInstances) {

        // System.out.println(Result.getResultAsString(result));
        // System.out.println(result.toString());

        double[][] predictions = result.allPredictions();
        StringBuilder builder = new StringBuilder();

        System.out.println("\n-- Prediction indexes --");
        for (double[] thisInstance : predictions) {
            for (double thisAttribute : thisInstance) {
                builder.append("[" + thisAttribute + "]");
            }
            System.out.println(builder.toString());
            builder = new StringBuilder();
        }

        System.out.println("\n-- Prediction values --");
        int instanceIndex = 0;
        for (double[] thisInstance : predictions) {

            int attributeIndex = 0;
            for (double thisAttribute : thisInstance) {

                builder.append(
                        "[" + attributeValue(predictInstances, instanceIndex, attributeIndex, thisAttribute) + "]");
                attributeIndex++;
            }
            System.out.println(builder.toString());
            builder = new StringBuilder();
            instanceIndex++;
        }
    }

    private String attributeValue(Instances instances, int instanceIndex, int attributeIndex, double valueIndex) {

        Instance thisInstance = instances.instance(instanceIndex);

        Attribute att = thisInstance.attribute(attributeIndex);

        String myValue = att.value((int) valueIndex);
        return myValue;
    }

    private void createPredictInstance(String temp, String cond, String enrgy, String gndr, Instances instances) {

        Instance instance = new DenseInstance(8);

        if (temp != null)
            instance.setValue(temperature, temp);
        if (cond != null)
            instance.setValue(conditions, cond);
        if (energy != null)
            instance.setValue(energy, enrgy);
        if (gender != null)
            instance.setValue(gender, gndr);

        instances.add(instance);

        // System.out.println(instance);
    }

    private Instances getInstances() throws Exception {

        InputStream stream = this.getClass().getClassLoader()
                .getResourceAsStream("weatherconditions.arff");

        Instances instance = DataSource.read(stream);
        MLUtils.prepareData(instance);

        return instance;

    }

    private Instances filterSomeData(Instances instances) {

        System.out.println(" -- before removing: " + instances.size());

        RemoveWithValues filter = new RemoveWithValues();

        String[] options = { "-C", "5", "-L", "1" };

        try {

            filter.setInputFormat(instances);
            filter.setOptions(options);
            Instances newInstances = Filter.useFilter(instances, filter);

            System.out.println(" -- after" + newInstances.size());
            return newInstances;

        } catch (Exception e) {
            e.printStackTrace();
            return instances;
        }

    }
}
