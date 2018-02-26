/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package research.crowdsourcing;

import ceka.consensus.ds.DawidSkene;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
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
        Clusterer[] clusterers = new Clusterer[10];
        Random rand = new Random(0);
        //Initialize all the clustering algorithms as K-means clusterers
        //with 2 centroids, which are randomly generated each time
        for(int i = 0; i < 10; i++){
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
           for(Clusterer clusterer : clusterers){
               
                Normalize normalize = new Normalize();
                normalize.setInputFormat(workerDataset);
                normalize.useFilter(workerDataset, normalize);
               clusterer.buildClusterer(workerDataset);
               for(int i = 0; i < workerDataset.getExampleSize(); i++){
                   
               }
           }
        }
        
        //TODO: Run each clusterer on this data set and extract each pair of clusters.
        
        //TODO: Use an ensemble technique to (1) determine which cluster is the
        //non-spammers and which cluster is the spammers, for each run of the clusterer,
        //and (2) integrate all the information together to reach a final conclusion
        //about each worker regarding whether he is a spammer.
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
