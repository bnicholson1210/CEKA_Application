/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package research.crowdsourcing;

import ceka.consensus.ds.DawidSkene;
import clustering.KMeans;
import weka.clusterers.HierarchicalClusterer;
import weka.core.Instances;
import ceka.core.Dataset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import research.ResultMetrics;
import stat.StatCalc;

public class JI_Cluster_Test {
    public static void main(String[] args) throws Exception{
        ArrayList<Dataset> datasets = Datasets.getCrowdsourcingDatasets();
        double avgAccAF = 0, avgAccBF = 0, avgAucAF = 0, avgAucBF = 0;

        for(Dataset d : datasets){
            System.out.println(d.relationName());
        }
        for (Dataset d : datasets){
            System.out.println("\nLooking at dataset " + d.relationName());
            Dataset filteredDataset = JIHeirarchicalClusterer(d);
            
            new DawidSkene(30).doInference(d);
            System.out.println("Acc before: "+ResultMetrics.accuracy(d));
            System.out.println("AUC before: "+ResultMetrics.auc(d));
            avgAccBF += ResultMetrics.accuracy(d);
            avgAucBF += ResultMetrics.auc(d);
            
            new DawidSkene(30).doInference(filteredDataset);
            System.out.println("Acc after: "+ResultMetrics.accuracy(filteredDataset));
            System.out.println("AUC after: "+ResultMetrics.auc(filteredDataset));
            avgAccAF += ResultMetrics.accuracy(filteredDataset);
            avgAucAF += ResultMetrics.auc(filteredDataset);
        }
        
        avgAccBF = avgAccBF/datasets.size();
        avgAucBF = avgAucBF/datasets.size();
        avgAccAF = avgAccAF/datasets.size();
        avgAucAF = avgAucAF/datasets.size();

        System.out.println("\nAvg Accuracy before filter: " + avgAccBF);
        System.out.println("Avg AUC before filter: " + avgAucBF);
        System.out.println("Avg Accuracy after filter: " + avgAccAF);
        System.out.println("Avg AUC after filter: " + avgAucAF);
        
    }
    
    public static Dataset JIHeirarchicalClusterer(Dataset dataset) throws Exception{
        

        WorkerTaskGraph graph = new WorkerTaskGraph(dataset);
        HashMap <String,Integer> commonLabels = graph.getMostCommonLabels();
        HashMap <String,Integer> uncommonLabels = graph.getLeastCommonLabels();
        int numOfTasks = graph.getTasks().size();
        
        // Initializing Worker ArrayLists used
        ArrayList<AnalyzedWorker> workers = graph.getWorkers();
        ArrayList<AnalyzedWorker> workers_to_test = new ArrayList();
        ArrayList<AnalyzedWorker> spammers = new ArrayList();


        //Keeps up with how many times a workers response matchs the most common response
        HashMap<String, Double> most_IC = new HashMap();
        HashMap<String, Double> least_IC = new HashMap();
        //Keeps up with how many unique responses their are between the worker and common response list
        HashMap<String, Double> unionCounts = new HashMap();

        // Collecting an (Intersection) set of the number of times a worker responded to a task
        //  with the same response as the most common/least response
        // Iterating over the tasks
        for(int n = 0; n<numOfTasks; n++){
            AnalyzedTask task = graph.getTasks().get(n);
            ArrayList<AnalyzedWorker> associatedWorkers = graph.allWorkersForTask(task);
//                System.out.println("most common: "+ commonLabels.get(task.getId()));
//                System.out.println("least common: "+ uncommonLabels.get(task.getId()));

            // Iterating over the workers who responded to this task
            for(int k=0; k < associatedWorkers.size(); k++){
                AnalyzedWorker w = associatedWorkers.get(k);
                int response = graph.labelFor(w, task);
                
                // Getting most/least common responses for this task
                Integer most_common_label = commonLabels.get(task.getId());
                Integer least_common_label = uncommonLabels.get(task.getId());

                // Initializing intersection counts
                Double most_icount = most_IC.get(""+w.getId());
                Double least_icount = least_IC.get(""+w.getId());
                if(most_icount == null) 
                    most_IC.put(""+w.getId(), new Double(0));
                if(least_icount == null) 
                    least_IC.put(""+w.getId(), new Double(0));

                // Intersection count if response aligns with most common label
                if(response == most_common_label){
                    if(most_icount == null) 
                        most_IC.put(""+w.getId(), new Double(1));
                    else 
                        most_IC.put(""+w.getId(), new Double(most_icount + 1));

                    Double ucount = unionCounts.get(""+w.getId());
                    if(ucount == null) 
                        unionCounts.put(""+w.getId(), new Double(1));                
                    else 
                        unionCounts.put(""+w.getId(), new Double(ucount + 1));
                }

                // Intersection count if response aligns with least common label
                else if(response == least_common_label){
                    if(least_icount == null) 
                        least_IC.put(""+w.getId(), new Double(1));
                    else 
                        least_IC.put(""+w.getId(), new Double(least_icount + 1));

                    Double ucount = unionCounts.get(""+w.getId());
                    if(ucount == null) 
                        unionCounts.put(""+w.getId(), new Double(1));                
                    else 
                        unionCounts.put(""+w.getId(), new Double(ucount + 1));
                }

                // Union count gets 2 added to it, one for the worker's response
                //  the other for the common response set
                else{
                    Double ucount = unionCounts.get(""+w.getId());
                    if(ucount == null) 
                        unionCounts.put(""+w.getId(), new Double(2));                
                    else 
                        unionCounts.put(""+w.getId(), new Double(ucount + 2));
                }
            }
        }
        
        
        
        
        // Keeping a map of the worker to most_JIndex pairs for lookup later
        Map<String, Double> IndexMap = new HashMap();
        
        // Calculating the Jaccard Index values, saving them and the number of
        //  tasks a worker went through in the data double array
        // data: [ [most_JIndex, NumOfTasks, least_JIndex], ...]
        double [][]data = new double[workers.size()][3];
        for(int n = 0; n < workers.size(); n++){
            AnalyzedWorker w = workers.get(n);
            try{
                //Calculate the Jaccard Index for a given worker
                double most_JIndex = most_IC.get(w.getId()) / unionCounts.get(w.getId());
                double least_JIndex = least_IC.get(w.getId()) / unionCounts.get(w.getId());

                IndexMap.put(""+w.getId(), most_JIndex);
                data[n][0] = most_JIndex;
                data[n][1] = graph.getNumTasksForWorker(w);
                data[n][2] = least_JIndex;
                
                // In the case of an Exception, we don't want to include the worker
                //  in the rest of the clusterer. That's why we're creating a new
                //  worker ArrayList filled with workers to test on the clusterer.
                workers_to_test.add(w);
                // This was necessary in variations of the algorithm where some
                //  workers didn't get a Jaccard Index score.
            }
            catch(Exception e){
                //No index recorded
                System.out.println("exception found!");
            }
        }

        
        
        
        // The Weka Clusters require Instances as input, we have a method in the
        //  KMeans class that converts a double array into Intances
        // We only call KMeans to serve as a way to structure the data appropriately
        KMeans handler = new KMeans();
        Instances data_inst = handler.createMultiDimData(data);
        
        int numData = data.length;
        int dimension = data[0].length;
        // This will hold the attributes for a worker along with the cluster
        //  result: [ [most_JIndex, NumOfTasks, least_JIndex, cluster], ...]
        double[][] result = new double[numData][dimension + 1];

        // Building our clusterer on the data
        int numClusters = 4;
        HierarchicalClusterer clusterer = new HierarchicalClusterer();
        clusterer.setNumClusters(numClusters);
        clusterer.buildClusterer(data_inst);

        // Saving the clusterer output to the result double array
        for(int i = 0; i < data_inst.numInstances(); i++)
        {
            result[i][dimension] = clusterer.clusterInstance(data_inst.instance(i));
            for(int j = 0; j < dimension; j++)
            {
                result[i][j] = data_inst.instance(i).value(j);
            }
        }


        
        
        // Now that the workers have been scored in different clusters
        //  we will make different ArrayLists of the cluster groups
        //  so we can take the average of their most common Jaccard Index
        ArrayList<AnalyzedWorker> one_workers = new ArrayList<AnalyzedWorker>();
        ArrayList<AnalyzedWorker> two_workers = new ArrayList<AnalyzedWorker>();
        ArrayList<AnalyzedWorker> three_workers = new ArrayList<AnalyzedWorker>();
        ArrayList<AnalyzedWorker> four_workers = new ArrayList<AnalyzedWorker>();

        double one_total_ji = 0.0;
        double two_total_ji = 0.0;
        double three_total_ji = 0.0;
        double four_total_ji = 0.0;
        int one_count = 0;
        int two_count = 0;
        int three_count = 0;
        int four_count = 0;
        // The order of instances from the clusterer result is the same as the order of our worker ArrayList
        for(int n = 0; n < workers_to_test.size(); n++){
            AnalyzedWorker w = workers_to_test.get(n);
            double w_ji = IndexMap.get(""+w.getId());
            int w_labels = graph.getNumTasksForWorker(w);
            int k=-1;
            double temp = 1000.0;
            int temp2=0;
            while(w_ji!=temp || w_labels != temp2){
                k++;
                temp = result[k][0];
                temp2 = (int)result[k][1];
                if(k>=10000000){
                    System.out.println("something's wrong");
                    System.exit(1);
                }
            }
            double cluster = result[k][3];
            if(cluster == 0.0){
                one_workers.add(w);
                one_total_ji += w_ji;
                one_count++;
            }
            else if(cluster == 1.0){
                two_workers.add(w);
                two_total_ji += w_ji;
                two_count++;
            }
            else if(cluster == 2.0){
                three_workers.add(w);
                three_total_ji += w_ji;
                three_count++;
            }
            else{
                four_workers.add(w);
                four_total_ji += w_ji;
                four_count++;
            }
        }
        
        
        
        
        // Look at the clusters and see which one contains the lowest
        //  most common Jaccard Index Score (this group will be the spammers)
        double one_avg = one_total_ji / one_count;
        double two_avg = two_total_ji / two_count;
        double three_avg = three_total_ji / three_count;
        double four_avg = four_total_ji / four_count;
        if(one_avg < two_avg && one_avg < three_avg && one_avg < four_avg){
            spammers = one_workers;
            int total = three_workers.size() + two_workers.size() + four_workers.size();
            System.out.println("clear count: "+total);
        }
        else if(two_avg < one_avg && two_avg < three_avg && two_avg < four_avg){
            spammers = two_workers;
            int total = one_workers.size() + three_workers.size() + four_workers.size();

            System.out.println("clear count: "+total);
        }
        else if(three_avg < one_avg && three_avg < two_avg && three_avg < four_avg){
            spammers = three_workers;
            int total = one_workers.size() + two_workers.size() + four_workers.size();
            System.out.println("clear count: "+total);
        }
        else{
            spammers = four_workers;
            int total = one_workers.size() + two_workers.size() + three_workers.size();
            System.out.println("clear count: "+total);
        }

        
        
        System.out.println("spammer count: "+spammers.size());
        return graph.removeSpammers(spammers);
    }
}
