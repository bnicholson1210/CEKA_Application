/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package research.crowdsourcing;

import ceka.consensus.MajorityVote;
import ceka.consensus.ds.DawidSkene;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import stat.StatCalc;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Bryce
 */
public class Filters 
{
        public static ArrayList<AnalyzedWorker> dynamicClassificationSpammerPicker(Dataset dataset, String name, 
                ArrayList<String> names, ArrayList<Dataset> datasets, ArrayList<String> attributeNames, 
                String evalAttribute, double filterLevel, Classifier classifier) throws Exception {
            ArrayList<Dataset> dat = new ArrayList();
            ArrayList<AnalyzedWorker> spammers = new ArrayList();
            dat.add(dataset);
            String thisDatasetArffFilename = Analysis.createWorkerAttributesArff(dat, attributeNames);
            Dataset workers = FileLoader.loadFile(thisDatasetArffFilename);
            Dataset workersCopy = FileLoader.loadFile(thisDatasetArffFilename);
            ArrayList<AnalyzedWorker> possibleSpammers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
            ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset, possibleSpammers, tasks);
            
            //printCorrelationCode(dataset);
            //workersCopy.setClassIndex(-1);
            Normalize normalize = new Normalize();
            normalize.setInputFormat(workersCopy);
            normalize.useFilter(workersCopy, normalize);
            
            boolean ready = false;
            double prop = .5;
            double adder = .25;
            int[] classifications = new int[0];
            double spamEst = 0;
            double spamPropEst = 0;
            int totalLabels = graph.getTotalNumLabels();
            int iterations = 0;
            double desiredProp = filterLevel;
            int desiredIterations = -1;
            double[] iterationVals = new double[10];
            //flag for starting over and stopping at detected iteration
            boolean flag = false;
            while(!ready)
            {
                if(iterations == 10)
                {
                    //System.out.println("Starting Over");
                    iterations = 0;
                    flag = true;
                    prop = .5;
                    adder = .25;
                }
                if(flag)
                {
                    double min = Double.POSITIVE_INFINITY;
                    int minIndex = 0;
                    for(int i = 0; i < iterationVals.length; i++)
                    {
                        if(Math.abs(desiredProp - iterationVals[i]) < min)
                        {
                            min = Math.abs(desiredProp - iterationVals[i]);
                            minIndex = i;
                        }
                    }
                    desiredIterations = minIndex;
                }
                
                Dataset others = Analysis.createSpammerDatasetFromAllOtherDatasets(datasets, names, name, attributeNames, evalAttribute, prop);
                classifier.buildClassifier(others);
                spamEst = 0;
                spamPropEst = 0;
                classifications = new int[workersCopy.getExampleSize()];
                //workersCopy.setClassIndex(3);
                //workersCopy.setClassIndex(5);
                workersCopy.setClassIndex(workersCopy.numAttributes() - 1);
                for(int i = 0; i < workersCopy.getExampleSize(); i++)
                {
                    Example e = workersCopy.getExampleByIndex(i);
                    int classification = (int)classifier.classifyInstance(e);
                    classifications[i] = classification;
                    if(classifications[i] == 1) {
                        spamPropEst += (double)possibleSpammers.get(i).getMultipleNoisyLabelSet(0).getLabelSetSize() /
                                (double)totalLabels;
                        spamEst++;
                    }
                }
                spamEst /= (double)workersCopy.getExampleSize();
                //System.out.println("Estimated proportion of spammers: " + spamEst);
                //System.out.println("Estimated proportion of labels: " + spamPropEst);
                iterationVals[iterations] = spamPropEst;
                if(iterations == desiredIterations)
                {
                    break;
                }
                if(spamPropEst < desiredProp - .05)
                {
                    prop += adder;
                    adder /= 2.0;
                }
                else if(spamPropEst > desiredProp + .05)
                {
                    prop -= adder;
                    adder /= 2.0;
                }
                else
                    ready = true;
                iterations++;
            }
            for(int i = 0; i < classifications.length; i++)
            {
                if(classifications[i] == 1)
                    spammers.add(possibleSpammers.get(i));
            }
            return spammers;
        }
        public static Dataset dynamicClassificationFiltering(Dataset dataset, String name, 
                ArrayList<String> names, ArrayList<Dataset> datasets, ArrayList<String> attributeNames, 
                String evalAttribute, double filterLevel, Classifier classifier) throws Exception
        {
            ArrayList<AnalyzedWorker> spammers = dynamicClassificationSpammerPicker(dataset,
                    name, names, datasets, attributeNames, evalAttribute, filterLevel, classifier);
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            return graph.removeSpammers(spammers);
        }
        
        public static ArrayList<AnalyzedWorker> RYSpammerPicker(Dataset dataset){
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            ArrayList<AnalyzedWorker> workers = graph.getWorkers();
            ArrayList<AnalyzedWorker> spammers = new ArrayList();            
            for(int i = 0; i < workers.size(); i++){
                AnalyzedWorker w = workers.get(i);
                double[] spammerScores = graph.spammerScore(w, null);
                //System.out.println("" + i + ": " + spammerScores[0] + "," + spammerScores[1] + ". " + (spammerScores[0] <= spammerScores[1] ? "Yes" : "No"));
                if(spammerScores[0] <= spammerScores[1]) 
                    spammers.add(w);
            }
            return spammers;
        }
        
        public static Dataset RYFilter(Dataset dataset){
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            ArrayList<AnalyzedWorker> spammers = RYSpammerPicker(dataset);
            return graph.removeSpammers(spammers);
        }
       
        public static ArrayList<AnalyzedWorker> IPWSpammerPicker(Dataset dataset){
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            ArrayList<AnalyzedWorker> workers = graph.getWorkers();
            ArrayList<AnalyzedWorker> spammers = new ArrayList();            
            for(int i = 0; i < workers.size(); i++){
                AnalyzedWorker w = workers.get(i);
                double[] workerCosts = graph.workerCost(w, null);
                //System.out.println("" + i + ": " + workerCosts[0] + "," + workerCosts[1] + ". " + (workerCosts[0] >= .5 * workerCosts[1] ? "Yes" : "No"));
                if(workerCosts[0] >= .5 * workerCosts[1]) 
                    spammers.add(w);
            }
            return spammers;
        }
                
        public static Dataset IPWFilter(Dataset dataset){
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            ArrayList<AnalyzedWorker> spammers = IPWSpammerPicker(dataset);
            return graph.removeSpammers(spammers);
        }
        

        
        public static ArrayList<AnalyzedWorker> CSNFSpammerPicker(Dataset dataset){
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            return graph.returnWorkersWithNoSimilarityPastNthQuartile(2);
        }
        
        public static Dataset CSNFilter(Dataset dataset){
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            ArrayList<AnalyzedWorker> spammers = CSNFSpammerPicker(dataset);
            return graph.removeSpammers(spammers);
        } 
        
        public static Dataset ensembleFilter(Dataset dataset, ArrayList<ArrayList<AnalyzedWorker>> spammerSets, int min){
            ArrayList<AnalyzedWorker> spammers = new ArrayList();
            HashMap<String, Object[]> spammerCounts = new HashMap();
            for(int i = 0; i < spammerSets.size(); i++){
                ArrayList<AnalyzedWorker> theseSpammers = spammerSets.get(i);
                for(int k = 0; k < theseSpammers.size(); k++){
                    String thisSpammerId = theseSpammers.get(k).getId();
                    if(spammerCounts.get(thisSpammerId) == null){
                        spammerCounts.put(thisSpammerId, new Object[]{1, theseSpammers.get(k)});
                    }
                    else{
                        Object[] arr = spammerCounts.get(thisSpammerId);
                        arr[0] = (Integer)arr[0] + 1;
                    }
                }
            }
            Iterator<String> it = spammerCounts.keySet().iterator();
            while(it.hasNext()){
                String key = it.next();
                if((Integer)spammerCounts.get(key)[0] >= min)
                    spammers.add((AnalyzedWorker)spammerCounts.get(key)[1]);
            }
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            return graph.removeSpammers(spammers);
        }
}

