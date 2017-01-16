import weka.core.*;
import libsvm.*;
import weka.core.converters.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.SnowballStemmer;
import weka.classifiers.trees.*;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.filters.*;
import weka.filters.unsupervised.attribute.*;
import java.io.*;
import java.util.*;

import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.rules.*;
import weka.classifiers.trees.*;
import weka.filters.supervised.attribute.*;

public class Categorization {
	
	public static Instances[][] split(Instances data, int numberOfFolds)
	{
		Instances [][] splitSet = new Instances[2][numberOfFolds];
		
		for(int i=0; i<numberOfFolds; i++)
		{
			splitSet[0][i] = data.trainCV(numberOfFolds, i);
			splitSet[1][i] = data.testCV(numberOfFolds, i);
		}
		
		return splitSet;
	}
	
	private static String fileToStr(String filepath) throws FileNotFoundException, IOException
	{
		BufferedReader br = new BufferedReader(new FileReader(filepath));
		StringBuffer fileContents = new StringBuffer();
		String line = br.readLine();
		while (line != null) {
			fileContents.append(line);
			line = br.readLine();
		}
		
		br.close();
		
		return fileContents.toString();
	}

	public static void main(String[] args) throws Exception {
		if (args.length != 2 && args.length != 3) {
			System.out.println("Usage");
			System.out.println("---------------");
			System.out.println("Comparision use: java -jar Categorization.jar [training folder location] [testing ARFF file]");
			System.out.println("NOTE: Testing files must be in ARFF format (.arff)\n");
			System.out.println("Statistical comparisons: java -jar Categorization.jar [training folder location] [testing folder location] [fold count]");
			System.out.println("NOTE: All categorization folders must be in the following format: /folder/category/file");
			System.exit(0);
		}
		
		
		if (args[1].matches(".*.arff")) {
			predictFile(args[0], args[1]);
		} else {
			if (args.length != 3) {
				System.out.println("Fold count needed. Terminating...");
				System.exit(0);
			}
			
			
			Instances trainSet = buildInstance(args[0]);
			Instances testSet = buildInstance(args[1]);
			
			Classifier[] classifiers = {new NaiveBayes(), new DecisionTable(), new DecisionStump()};
			for (int i=0; i<classifiers.length; i++) {
				Evaluation eval = new Evaluation(trainSet);
				classifiers[i].buildClassifier(trainSet);
				eval.crossValidateModel(classifiers[i], testSet, Integer.parseInt(args[2]), new Random(1));
				
				System.out.println("Performance results for " + classifiers[i].getClass().getSimpleName() 
						+ "\n---------------------------------");

				System.out.println(eval.toSummaryString("\nResults",true));
				System.out.println("fmeasure: " +eval.fMeasure(1) + " Precision: " + eval.precision(1)+ " Recall: "+ eval.recall(1));

				System.out.println(eval.toMatrixString());

				System.out.println(eval.toClassDetailsString());

				System.out.println("AUC = " +eval.areaUnderROC(1));
				System.out.println("--------------------------------------");
			}
			
		}
	}

	private static void predictFile(String trainPath, String testPath) throws Exception, IOException {
		Instances[] trainAndTest = buildInstances(trainPath, testPath);
		Instances trainData = convert(trainAndTest[0]);
		Instances testData = convert(trainAndTest[1]);
		
		Classifier[] classifiers = {new NaiveBayes(), new DecisionTable(), new DecisionStump()};
		System.out.println("Categorizing...");
		for (int i=0; i<classifiers.length; i++)
		{
			Classifier classifier = evaluateAndBuild(classifiers[i],trainData);
			
			Evaluation eval = new Evaluation(trainData);
			
			AddClassification addclass = new AddClassification();
			addclass.setClassifier(classifier);
			addclass.setInputFormat(testData);
			Instances output = Filter.useFilter(testData, addclass);
			
			double classified = classifier.classifyInstance(output.instance(0));
			
			System.out.print(classifier.getClass() + ": ");
			switch ((int)classified) {
			case 0:
				System.out.println("FAQ");
				break;
			case 1:
				System.out.println("Forum");
				break;
			case 2:
				System.out.println("Internal KnowledgeBase");
				break;
			case 3:
				System.out.println("KnowledgeBase");
				break;
			case 4:
				System.out.println("Main documentation");
				break;
			}
		}
	}
	
	private static Instances buildInstance(String trainPath) throws Exception
	{
		TextDirectoryLoader loader = new TextDirectoryLoader();		
		loader.setDirectory(new File(trainPath));
		Instances rawData = loader.getDataSet();
		
		Instances filteredData = convert(rawData);
		return filteredData;
	}
	
	private static Instances[] buildInstances(String trainPath, String testPath) throws Exception
	{
		//Get train instance
		TextDirectoryLoader loader = new TextDirectoryLoader();		
		loader.setDirectory(new File(trainPath));
		Instances rawData = loader.getDataSet();
		
		//Get test instance
		createArff(testPath);
		DataSource testSource = new DataSource(testPath);
		Instances instances = testSource.getDataSet();
		
		//Standardize
		Standardize strd = new Standardize();
		strd.setInputFormat(rawData);
		Instances newTrain = Filter.useFilter(rawData, strd);
		Instances newTest = Filter.useFilter(instances, strd);
		
		Instances[] outInstances = {newTrain, newTest};
		return outInstances;
		
	}

	private static Classifier evaluateAndBuild(Classifier classifier, Instances instances) throws IOException,
			Exception {
		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, 4, new Random(1));
		classifier.buildClassifier(instances);

		return classifier;
	}
	
	private static Instances convert(Instances unfiltered) throws Exception
	{
		SnowballStemmer stemmer = new SnowballStemmer();
		stemmer.setStemmer("english");
		StringToWordVector filter = new StringToWordVector();
		filter.setStemmer(stemmer);
		filter.setInputFormat(unfiltered);

		Instances filteredData = Filter.useFilter(unfiltered, filter);
		
		return filteredData;
	}

	private static void createArff(String filepath) throws IOException
	{
		FileWriter arff;
		File dir = new File(".");
		String currentDir = dir.getAbsolutePath();
		
		arff = new FileWriter(currentDir + "/arff.arff", false);
		
		arff.write("@relation test\n\n");
		arff.write("@attribute text string\n");
		arff.write("@attribute @@class@@ {faq,forum,ikb,kb,maindocs}\n\n");
		arff.write("@data\n\n");
		arff.write("'" + fileToStr(filepath) + "',kb\n");
		
		arff.close();
	}
}
	
