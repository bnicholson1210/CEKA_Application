/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package research;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.utils.DatasetManipulator;
import ceka.utils.PerformanceStatistic;
import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Classifier;

public class ResultMetrics {
    
    public static double accuracy(Dataset dataset)
    {
        int correct = 0;
        for(int i = 0; i < dataset.getExampleSize(); i++)
        {
            Example e = dataset.getExampleByIndex(i);
            if(e.getIntegratedLabel().getValue() == e.getTrueLabel().getValue())
                correct++;
        }
        return (double)correct / (double)dataset.getExampleSize();
    }

    public static double auc(Dataset dataset)
    {
        PerformanceStatistic ps = new PerformanceStatistic();
        ps.stat(dataset);
        return ps.getAUC();
    }
    
    public static double validation(Dataset dataset, Classifier classifier, int numFolds) throws Exception
    {
        Dataset newDataset = DatasetUtils.makeCopy(dataset);
        newDataset.randomize(new Random());
        int spam = 0;
        int nonspam = 0;
        int total = 0;
        for(int i = 0; i < newDataset.getExampleSize(); i++)
        {
            Example e = newDataset.getExampleByIndex(i);
            if(e.getTrueLabel().getValue() == 0)
                nonspam++;
            else if (e.getTrueLabel().getValue() == 1)
                spam++;
            total++;  
        }

        System.out.println("Total examples: " + total);
        System.out.println("Nonspam examples: " + nonspam);
        System.out.println("Spam examples: " + spam);
        int spamCorrect = 0;
        int spamTotal = 0;
        int nonspamCorrect = 0;
        int nonspamIncorrect = 0;
        int spamIncorrect = 0;
        int nonspamTotal = 0;
        Dataset[] folds = DatasetManipulator.split(newDataset, numFolds, true);
        for(int i = 0; i < numFolds; i++)
        {
            Dataset[] trainAndTest = DatasetManipulator.pickCombine(folds, i);
            Dataset train = trainAndTest[0];
            Dataset test = trainAndTest[1];
            classifier.buildClassifier(train);
            for(int j = 0; j < test.getExampleSize(); j++)
            {
                Example e = test.getExampleByIndex(j);
                int label = e.getTrueLabel().getValue();
                if(label == 0)
                {
                    if(classifier.classifyInstance(e) == label)
                        nonspamCorrect++;
                    else
                        nonspamIncorrect++;
                    nonspamTotal++;
                }
                else if(label == 1)
                {
                    if(classifier.classifyInstance(e) == label)
                        spamCorrect++;
                    else
                        spamIncorrect++;
                    spamTotal++;
                }
            }
        }
        System.out.println("Nonspam:");
        System.out.println("\tPrecision: " + (double)nonspamCorrect / ((double)nonspamCorrect + (double)nonspamIncorrect));
        System.out.println("\tRecall: " + (double)nonspamCorrect / ((double)nonspamCorrect + spamIncorrect));

        System.out.println("Spam:");
        System.out.println("\tPrecision: " + (double)spamCorrect / ((double)spamCorrect + (double)spamIncorrect));
        System.out.println("\tRecall: " + (double)spamCorrect / ((double)spamCorrect + (double) nonspamIncorrect));
        return ((double)spamCorrect + (double)nonspamCorrect) / ((double)spamCorrect + 
                (double)nonspamCorrect + (double)spamIncorrect + (double)nonspamIncorrect);
    }
        
    private static double accuracy(ArrayList<Integer> intLabels, ArrayList<Integer> trueLabels){
            double acc = 0;
            int total = 0;
            int correct = 0;
            for(int i = 0; i < intLabels.size(); i++){
                total++;
                if(intLabels.get(i) == trueLabels.get(i))
                    correct++;
            }
            return (double)correct / (double)total;
        }
        
        private static double precision(ArrayList<Integer> intLabels, ArrayList<Integer> trueLabels){
            int truePositive = 0;
            int falsePositive = 0;
            int trueNegative = 0;
            int falseNegative = 0;
            
            for(int i = 0; i < intLabels.size(); i++){
                int intLabel = intLabels.get(i);
                int trueLabel = trueLabels.get(i);
                if(intLabel == trueLabel){
                    if(intLabel == 0){
                        trueNegative++;
                    }
                    else{
                        truePositive++;
                    }
                }
                else{
                    if(intLabel == 0){
                        falseNegative++;
                    }
                    else{
                        falsePositive++;
                    }
                }
            }
            return (double)truePositive / (double)(truePositive + falsePositive);
        }
        
        private static double recall(ArrayList<Integer> intLabels, ArrayList<Integer> trueLabels){
            int truePositive = 0;
            int falsePositive = 0;
            int trueNegative = 0;
            int falseNegative = 0;
            
            for(int i = 0; i < intLabels.size(); i++){
                int intLabel = intLabels.get(i);
                int trueLabel = trueLabels.get(i);
                if(intLabel == trueLabel){
                    if(intLabel == 0){
                        trueNegative++;
                    }
                    else{
                        truePositive++;
                    }
                }
                else{
                    if(intLabel == 0){
                        falseNegative++;
                    }
                    else{
                        falsePositive++;
                    }
                }
            }
            return (double)truePositive / (double)(truePositive + falseNegative);
        }
        
        private static double f1(ArrayList<Integer> intLabels, ArrayList<Integer> trueLabels){
            return 2.0 * (precision(intLabels, trueLabels) * recall(intLabels, trueLabels)) / 
                    (precision(intLabels, trueLabels) + recall(intLabels, trueLabels));
        }
}
