package research.crowdsourcing;

import ceka.consensus.ds.DawidSkene;
import ceka.core.Dataset;
import java.util.ArrayList;
import research.ResultMetrics;
import weka.classifiers.lazy.IBk;

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
        ArrayList<String> datasetNames = new ArrayList();
        for(Dataset dataset : datasets){
            datasetNames.add(dataset.relationName());
        }
        for(Dataset dataset : datasets){            
            //Dataset filteredDataset = Filters.IPWFilter(dataset);
            System.out.println("\n*************** LOOKING AT DATASET " + dataset.relationName()+" *************");
//            Dataset filteredDataset = Filters.JIFilter(dataset);
//            ArrayList<AnalyzedWorker> JIspammers = Filters.JISpammerPicker(dataset, null);
//            ArrayList<AnalyzedWorker> CSNFspammers = Filters.CSNFSpammerPicker(dataset);
//            ArrayList<AnalyzedWorker> IPWspammers = Filters.IPWSpammerPicker(dataset);
//            ArrayList<AnalyzedWorker> RYspammers = Filters.RYSpammerPicker(dataset);
//
//             ArrayList<String> attributeSet = new ArrayList();
//            attributeSet.add("distanceFromAverageEvenness");
//            attributeSet.add("logSimilarity");
//            attributeSet.add("spammerScore");
//            attributeSet.add("workerCost");
//            attributeSet.add("proportion");
//            String evaluationAttribute = "EMAccuracy";
//            new DawidSkene(30).doInference(dataset);
//            ArrayList<AnalyzedWorker> DCspammers = Filters.dynamicClassificationSpammerPicker(dataset,
//                    dataset.relationName(), datasetNames, datasets, attributeSet, evaluationAttribute, .5, new IBk(5), null);
//            
//            ArrayList<ArrayList<AnalyzedWorker>> spamList = new ArrayList();
//            spamList.add(DCspammers);
//            spamList.add(JIspammers);
//            spamList.add(CSNFspammers);
//            spamList.add(IPWspammers);
//            spamList.add(RYspammers);
//            
//            Dataset filteredDataset = Filters.ensembleFilter(dataset, spamList, 2);
//            
            Dataset filteredDataset = Filters.DJIFilter(dataset);
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
