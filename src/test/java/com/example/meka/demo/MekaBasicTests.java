package com.example.meka.demo;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import org.junit.Before;
import org.junit.Test;

import meka.classifiers.MultiXClassifier;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.core.Result;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;

public class MekaBasicTests {

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

	private Attribute watchFootball;
	private Attribute goSkiing;
	private Attribute goToPark;
	private Attribute readABook;
	private Attribute temperature;
	private Attribute conditions;
	private Attribute energy;

	private ArrayList<Attribute> attributes;

	private MultiXClassifier classifier;

	@Before
	public void setup() {

		List<String> yesOrNoList = new ArrayList<>();
		yesOrNoList.add(NO);
		yesOrNoList.add(YES);

		List<String> temperatureList = new ArrayList<>();
		temperatureList.add(COLD);
		temperatureList.add(WARM);
		temperatureList.add(HOT);

		List<String> weatherConditionsList = new ArrayList<>();
		weatherConditionsList.add(CLEAR);
		weatherConditionsList.add(RAIN);
		weatherConditionsList.add(SNOW);

		List<String> energyLevelList = new ArrayList<>();
		energyLevelList.add(LAZY);
		energyLevelList.add(ENERGETIC);

		// these are to be predicted based on conditions
		watchFootball = new Attribute("football", yesOrNoList);
		goSkiing = new Attribute("goskiing", yesOrNoList);
		goToPark = new Attribute("gotopark", yesOrNoList);
		readABook = new Attribute("readabook", yesOrNoList);

		// these are conditions, will predict the above activities
		temperature = new Attribute("temperature", temperatureList);
		conditions = new Attribute("conditions", weatherConditionsList);
		energy = new Attribute("energetic", energyLevelList);

		attributes = new ArrayList<>();
		attributes.add(watchFootball);
		attributes.add(goSkiing);
		attributes.add(readABook);
		attributes.add(goToPark);
		attributes.add(temperature);
		attributes.add(conditions);
		attributes.add(energy);

	}

	@Test
	public void simpleTests() throws Exception {

		int howMany = 1;

		Instances trainInstances = new Instances("WeatherConditions", attributes, 100);
		Instances predictInstances = new Instances("WhatToDo", attributes, 1);

		trainInstances.setClassIndex(4);
		predictInstances.setClassIndex(4);

		// .................go to the park - V
		// ...............read a book - V
		// .............go skiing - V
		// ....watch football - V
		createTrainingInstance(YES, NO, YES, NO, WARM, CLEAR, LAZY, trainInstances, howMany);
		createTrainingInstance(YES, NO, YES, YES, HOT, CLEAR, ENERGETIC, trainInstances, howMany);

		createPredictInstance(COLD, SNOW, ENERGETIC, predictInstances);

		Result result = getPrediction(trainInstances, predictInstances);

		assertPredictions(result, 0, true, false, false, true);

	}

	private void assertPredictions(Result result, int index, boolean football, boolean ski, boolean book,
			boolean park) {

		int[][] predictions = result.allPredictions(THRESHOLD);

		assertEquals("football result not expected", football ? 1 : 0, predictions[index][0]);

		// should have other assertions here, too

	}

	private Result getPrediction(Instances trainInstances, Instances predictInstances) throws Exception {

		classifier = new CC();

		Result result = Evaluation.evaluateModel(classifier, trainInstances, predictInstances);

		System.out.println(Result.getResultAsString(result));
		System.out.println(result.toString());

		return result;
	}

	private void createPredictInstance(String temp, String cond, String enrgy, Instances instances) {

		Instance instance = new DenseInstance(7);

		if (temp != null)
			instance.setValue(temperature, temp);
		if (cond != null)
			instance.setValue(conditions, cond);
		if (energy != null)
			instance.setValue(energy, enrgy);

		instances.add(instance);
	}

	private void createTrainingInstance(String fb, String ski, String book, String park, String temp, String cond,
			String enrgy, Instances instances, int howMany) {

		for (int x = 0; x < howMany; x++) {
			Instance instance = new DenseInstance(7);
			if (fb != null)
				instance.setValue(watchFootball, fb);
			if (ski != null)
				instance.setValue(goSkiing, ski);
			if (book != null)
				instance.setValue(readABook, book);
			if (park != null)
				instance.setValue(goToPark, park);
			if (temp != null)
				instance.setValue(temperature, temp);
			if (cond != null)
				instance.setValue(conditions, cond);
			if (energy != null)
				instance.setValue(energy, enrgy);

			instances.add(instance);
		}
	}

}
