package research.crowdsourcing;

import ceka.consensus.ds.DawidSkene;
import ceka.core.Dataset;
import java.util.ArrayList;
import research.ResultMetrics;

/**
 *
 * @author Nick
 */
public class JFilterTest {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception{
                
        ArrayList<Dataset> datasets = Datasets.getCrowdsourcingDatasets();
        double avgAccAF = 0, avgAccBF = 0, avgAucAF = 0, avgAucBF = 0;
        for(Dataset dataset : datasets){            
            //Dataset filteredDataset = Filters.IPWFilter(dataset);
            System.out.println("\n*************** LOOKING AT DATASET " + dataset.relationName()+" *************");
            Dataset filteredDataset = Filters.JIFilter(dataset);

            new DawidSkene(30).doInference(dataset);
            System.out.println("Accuracy before filter: " + ResultMetrics.accuracy(dataset));
            avgAccBF = avgAccBF + ResultMetrics.accuracy(dataset);
            System.out.println("AUC before filter: " + ResultMetrics.auc(dataset));
            avgAucBF = avgAucBF + ResultMetrics.auc(dataset);
            new DawidSkene(30).doInference(filteredDataset);
            System.out.println("Accuracy after filter: " + ResultMetrics.accuracy(filteredDataset));
            avgAccAF = avgAccAF + ResultMetrics.accuracy(filteredDataset);
            System.out.println("AUC after filter: " + ResultMetrics.auc(filteredDataset));
            avgAucAF = avgAucAF + ResultMetrics.auc(filteredDataset);
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
        
    
}
