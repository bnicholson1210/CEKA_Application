/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package research.crowdsourcing;

import java.util.Random;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;


public class MainExperiments {
    public static void main(String[] args) throws Exception{
        runExperiments();
    }
    
    public static void runExperiments() throws Exception{
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
        
        //TODO: Create dataset of workers where the (4) features are
        //from Dynamic Classification Filter, Cosine Similarity Neighborhood
        //Filter, RY Filter, and IPW filter. 
        
        //TODO: Run each clusterer on this data set and extract each pair of clusters.
        
        //TODO: Use an ensemble technique to (1) determine which cluster is the
        //non-spammers and which cluster is the spammers, for each run of the clusterer,
        //and (2) integrate all the information together to reach a final conclusion
        //about each worker regarding whether he is a spammer.
    }
}
