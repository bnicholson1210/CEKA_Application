/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package research.crowdsourcing;

import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.core.Example;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import stat.StatCalc;
import weka.classifiers.Classifier;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Bryce, Nick
 */
public class Filters
{
        public static ArrayList<AnalyzedWorker> dynamicClassificationSpammerPicker(Dataset dataset, String name,
                ArrayList<String> names, ArrayList<Dataset> datasets, ArrayList<String> attributeNames,
                String evalAttribute, double filterLevel, Classifier classifier, Map<String, Double> confs) throws Exception {
            if(confs == null) confs = new HashMap<>();
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
                confs.clear();
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
                    double[] distribution = classifier.distributionForInstance(e);
                    //System.out.println("Classification: " + classification + ", conf0: " + distribution[0] + ", conf1: " + distribution[1]);
                    if(classification==0){
                        confs.put(possibleSpammers.get(i).getId(), distribution[classification]);
                    }
                    else{
                      confs.put(possibleSpammers.get(i).getId(), (-distribution[classification]));
                    }
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
                String evalAttribute, double filterLevel, Classifier classifier, Map<String, Double> confs) throws Exception
        {
            ArrayList<AnalyzedWorker> spammers = dynamicClassificationSpammerPicker(dataset,
                    name, names, datasets, attributeNames, evalAttribute, filterLevel, classifier, confs);
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            return graph.removeSpammers(spammers);
        }

        public static Dataset dynamicClassificationFiltering(Dataset dataset, String name,
                ArrayList<String> names, ArrayList<Dataset> datasets, ArrayList<String> attributeNames,
                String evalAttribute, double filterLevel, Classifier classifier) throws Exception{
            return dynamicClassificationFiltering(dataset, name, names, datasets, attributeNames,
                    evalAttribute, filterLevel, classifier, null);
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

        public static ArrayList<AnalyzedWorker> DynamicJISpammerPicker(Dataset dataset, Map<String, Double> IndexMap){
            if (IndexMap == null) IndexMap = new HashMap();
            
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            HashMap <String,Integer> commonLabels = graph.getMostCommonLabels();
//            HashMap <String, Integer> commonLabels = graph.getKeyMap(); //use cuttoff 0.26
            int numOfTasks = graph.getTasks().size();
            ArrayList<AnalyzedWorker> workers = graph.getWorkers();
            ArrayList<AnalyzedWorker> spammers = new ArrayList();
            
            //Keeps up with how many times a workers response matchs the most common response
            HashMap<String, Double> interCounts = new HashMap();
            //Keeps up with how many unique responses their are between the worker and common response list
            HashMap<String, Double> unionCounts = new HashMap();

            for(int n = 0; n<numOfTasks; n++){
                AnalyzedTask task = graph.getTasks().get(n);
                ArrayList<AnalyzedWorker> associatedWorkers = graph.allWorkersForTask(task);
                for(int k=0; k < associatedWorkers.size(); k++){
                    AnalyzedWorker w = associatedWorkers.get(k);
                    int response = graph.labelFor(w, task);

                    Double icount = interCounts.get(""+w.getId());
                    if(icount == null) interCounts.put(""+w.getId(), new Double(0));
                    if(response == commonLabels.get(task.getId())){
                        if(icount == null) interCounts.put(""+w.getId(), new Double(1));
                        else interCounts.put(""+w.getId(), new Double(icount + 1));
                        
                        Double ucount = unionCounts.get(""+w.getId());
                        if(ucount == null) unionCounts.put(""+w.getId(), new Double(1));                
                        else unionCounts.put(""+w.getId(), new Double(ucount + 1));
                    }
                    else{
                        Double ucount = unionCounts.get(""+w.getId());
                        if(ucount == null) unionCounts.put(""+w.getId(), new Double(2));                
                        else unionCounts.put(""+w.getId(), new Double(ucount + 2));
                    }
                }
            }
            
            for(int n = 0; n < workers.size(); n++){
                AnalyzedWorker w = workers.get(n);
                try{
                    //Calculate the Jaccard Index for a given worker
                    double JIndex = interCounts.get(w.getId()) / unionCounts.get(w.getId());
                    IndexMap.put(""+w.getId(), JIndex);
                }
                catch(Exception e){
                    System.out.println("no index found for this guy "+w.getId());
                }
            }
            
            double []vals = new double[IndexMap.size()];
            int count = 0;
            Iterator<String> it = IndexMap.keySet().iterator();
            while(it.hasNext()){
                vals[count] = IndexMap.get(it.next());
                count++;
            }

            double averageJIndex = StatCalc.mean(vals);
//            double var = StatCalc.variance(vals);
//            double sigma = Math.sqrt(var); //Standard Dev
            double []quartiles = StatCalc.quartiles(vals);
            
            //Looking for the lowest JI value that isn't 0
            Arrays.sort(vals);
            double min = 0;
            int a = 0;
            //non-zero min
            while(min==0){
                if(vals[a]>0)
                    min=vals[a];
                a++;
            }
            
            double cuttoff = min + (averageJIndex - quartiles[0]);
//            System.out.println("Cuttoff is: "+cuttoff+" Mean: "+averageJIndex+ " sigma: "+sigma);
//            System.out.println("\n\nvals"+IndexMap.values());
            for(int n = 0; n < workers.size(); n++){
                AnalyzedWorker w = workers.get(n);
                if(IndexMap.get(""+w.getId()) <= cuttoff) 
                    spammers.add(w);
            }
            
            return spammers;
        }
        
        public static Dataset DJIFilter(Dataset dataset, Map<String, Double> JIndexes){
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            ArrayList<AnalyzedWorker> spammers = DynamicJISpammerPicker(dataset, JIndexes);
            return graph.removeSpammers(spammers);
        }
        public static Dataset DJIFilter(Dataset dataset){
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            ArrayList<AnalyzedWorker> spammers = DynamicJISpammerPicker(dataset, null);
            return graph.removeSpammers(spammers);
        }
        public static void avgNumberOfAnswers(Dataset dataset, ArrayList<AnalyzedWorker> workers){
            double result = 0.0, val;
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            for(int i = 0; i < workers.size(); i++){
                val = graph.allTasksForWorker(workers.get(i)).size();
                System.out.println("Worker "+(i+1)+" answered "+val+" tasks");
                result = result + val;
            }
            result = result / workers.size();
            System.out.println("\nAverage number of tasks answered by this group: "+result);
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
