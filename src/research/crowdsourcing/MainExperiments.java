/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package research.crowdsourcing;

import ceka.consensus.ds.DawidSkene;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.core.Example;
import cekax.utils.DatasetMapper;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import research.ResultMetrics;
import stat.StatCalc;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;


public class MainExperiments {
    public static void main(String[] args) throws Exception{
        runExperiments();
    }

    public static void runExperiments() throws Exception{
        ArrayList<Dataset> datasets = Datasets.getCrowdsourcingDatasets();
        //Set number of clustering algorithms that will be run
        int numClusterers = 10;
        Clusterer[] clusterers = new Clusterer[numClusterers];
        Random rand = new Random(0);
        //Initialize all the clustering algorithms as K-means clusterers
        //with 2 centroids, which are randomly generated each time
        for(int i = 0; i < numClusterers; i++){
            SimpleKMeans kmeans = new SimpleKMeans();
            kmeans.setNumClusters(2);
            kmeans.setSeed(rand.nextInt());
            clusterers[i] = kmeans;
        }
        //Store all the data set names in a list
        //(needed for DCF algorithm)
        ArrayList<String> datasetNames = new ArrayList();
        for(Dataset dataset : datasets){
            datasetNames.add(dataset.relationName());
        }
        for(Dataset dataset : datasets){
            //Create dataset of workers where the (4) features are
            //from Dynamic Classification Filter, Cosine Similarity Neighborhood
            //Filter, RY Filter, and IPW filter.
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
            ArrayList<AnalyzedWorker> workers = graph.getWorkers();
            double[] dcfData = new double[workers.size()];
            double[] csnfData = new double[workers.size()];
            double[] ryData = new double[workers.size()];
            double[] ipwData = new double[workers.size()];

            Map<String, Double> confs = new HashMap<>();
            ArrayList<String> attributeSet = new ArrayList();
            attributeSet.add("distanceFromAverageEvenness");
            attributeSet.add("logSimilarity");
            attributeSet.add("spammerScore");
            attributeSet.add("workerCost");
            attributeSet.add("proportion");
            String evaluationAttribute = "EMAccuracy";
            new DawidSkene(30).doInference(dataset);

            Filters.dynamicClassificationFiltering(dataset, dataset.relationName(), datasetNames,
                    datasets, attributeSet, evaluationAttribute, .5, new IBk(5), confs);

            for(int i = 0; i < workers.size(); i++){
                AnalyzedWorker worker = workers.get(i);
                dcfData[i] = confs.get(worker.getId());
                csnfData[i] = graph.getAverageSimilarityWithAllOtherWorkers(worker);
                ryData[i] = graph.getCharacteristicValueForWorker("spammerScore", worker);
                ipwData[i] = graph.getCharacteristicValueForWorker("workerCost", worker);
            }

           String workerDatasetArffFilename = createWorkerDataset(dataset.relationName(), dcfData, csnfData, ryData, ipwData);
           Dataset workerDataset = FileLoader.loadFile(workerDatasetArffFilename);
           workerDataset.setClassIndex(-1);
           DatasetMapper<AnalyzedWorker> datasetMapper = new DatasetMapper(workerDataset, workers);
           Map<AnalyzedWorker, List<Integer>> workerClusters = new HashMap<>();
           Map<AnalyzedWorker, List<Integer>> workerQualityClusters = new HashMap<>();
           Map<AnalyzedWorker, List<Double>> workerQualityClusterValues = new HashMap<>();
           WorkerTaskGraph wtg = new WorkerTaskGraph(dataset);
           final Integer GOOD_CLUSTER_CODE = 0;
           final Integer BAD_CLUSTER_CODE = 1;
           for(Clusterer clusterer : clusterers){
               Normalize normalize = new Normalize();
               normalize.setInputFormat(workerDataset);
               normalize.useFilter(workerDataset, normalize);
               clusterer.buildClusterer(workerDataset);
               for(int i = 0; i < workerDataset.getExampleSize(); i++){
                   Example e = workerDataset.getExampleByIndex(i);
                   int clusterNumber = clusterer.clusterInstance(e);
                   AnalyzedWorker w = datasetMapper.getAssociatedObjectOfExample(e);
                   List<Integer> thisWorkerClusters = workerClusters.get(w);
                   if(thisWorkerClusters == null){
                       thisWorkerClusters = new ArrayList<>();
                       workerClusters.put(w, thisWorkerClusters);
                   }
                   thisWorkerClusters.add(clusterNumber);
               }
           }
           for(int i = 0; i < clusterers.length; i++){
               List<Double> cluster0accs = new ArrayList<>();
               List<Double> cluster1accs = new ArrayList<>();
               Integer thisBadCluster;
               Integer thisGoodCluster;
               for(int j = 0; j < workers.size(); j++){
                    AnalyzedWorker w = workers.get(i);
                    Double workerEmAccuracy = wtg.getCharacteristicValueForWorker("EMAccuracy", w);
                    if(workerClusters.get(w).get(i) == 0){
                        cluster0accs.add(workerEmAccuracy);
                    }else{
                        cluster1accs.add(workerEmAccuracy);
                    }
               }
               Double cluster0Mean = StatCalc.mean(cluster0accs);
               Double cluster1Mean = StatCalc.mean(cluster1accs);
               if(cluster0Mean >= cluster1Mean){
                   thisGoodCluster = 0;
                   thisBadCluster = 1;
               }else{
                   thisGoodCluster = 1;
                   thisBadCluster = 0;
               }
               for(int j = 0; j < workers.size(); j++){
                   AnalyzedWorker w = workers.get(j);
                   List<Integer> thisWorkerQualityClusters = workerQualityClusters.get(w);
                   List<Double> thisWorkerQualityClusterValues = workerQualityClusterValues.get(w);
                   if(thisWorkerQualityClusters == null){
                       thisWorkerQualityClusters = new ArrayList<>();
                       workerQualityClusters.put(w, thisWorkerQualityClusters);
                   }
                   if(thisWorkerQualityClusterValues == null){
                       thisWorkerQualityClusterValues = new ArrayList<>();
                       workerQualityClusterValues.put(w, thisWorkerQualityClusterValues);
                   }
                   thisWorkerQualityClusters.add((thisGoodCluster == workerClusters.get(w).get(i) ? GOOD_CLUSTER_CODE : BAD_CLUSTER_CODE));
               }
           }
           List<AnalyzedWorker> spammers = new ArrayList<>();
           /*for(int i = 0; i < workers.size(); i++){
               AnalyzedWorker w = workers.get(i);
               List<Integer> thisWorkerQualityClusters = workerQualityClusters.get(w);
               Double sum = 0.0;
               Double total = 0.0;
               for(int j = 0; j < thisWorkerQualityClusters.size(); j++){
                   total++;
                   if(thisWorkerQualityClusters.get(j) == BAD_CLUSTER_CODE){
                       sum++;
                   }
               }
               if(sum / total > .5){
                   spammers.add(w);
               }
           }*/
           for(int i = 0; i < workers.size(); i++){
               AnalyzedWorker w = workers.get(i);
               List<Double> thisWorkerQualityClusterValues = workerQualityClusterValues.get(w);
               Double total = 0.0;
               for(int j = 0; j < thisWorkerQualityClusterValues.size(); j++){
                   total += thisWorkerQualityClusterValues.get(j);
               }
               if(total < 0){
                   spammers.add(w);
               }

           }
           Dataset filteredDataset = wtg.removeSpammers(spammers);
           new DawidSkene(30).doInference(dataset);
           System.out.println("Accuracy before filter: " + ResultMetrics.accuracy(dataset));
           System.out.println("AUC before filter: " + ResultMetrics.auc(dataset));
           new DawidSkene(30).doInference(filteredDataset);
           System.out.println("Accuracy after filter: " + ResultMetrics.accuracy(filteredDataset));
           System.out.println("AUC after filter: " + ResultMetrics.auc(filteredDataset));
        }
    }

    private static String createWorkerDataset(String datasetName, double[] dcfData, double[] csnfData,
            double[] ryData, double[] ipwData) throws Exception{
        String fn;
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(
                fn = "/customDatasets/workerEnsembleClustering/" + datasetName + ".arff")));
        bw.write("@relation\t" + datasetName + "\n");
        bw.write("@attribute\tatt1\treal\n");
        bw.write("@attribute\tatt2\treal\n");
        bw.write("@attribute\tatt3\treal\n");
        bw.write("@attribute\tatt4\treal\n");
        bw.write("@attribute\tclass\t{0,1}\n");

        bw.write("@data\n");

        for(int i = 0; i < dcfData.length; i++){
            bw.write("" + dcfData[i] + "," + csnfData[i] + "," + ryData[i] + "," + ipwData[i] + ",0\n");
        }

        bw.close();

        return fn;
    }
}
