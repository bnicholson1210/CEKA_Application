/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package research.crowdsourcing;

import ceka.converters.FileLoader;
import ceka.core.Dataset;
import java.util.ArrayList;

public class Datasets {
    public static final String basePath = "data\\real-world\\";
    public static final String eastpolitics2000Path = basePath + "eastpolitics\\eastpolitics2000AMT\\";
    public static final String eastpolitics1000Path = basePath + "eastpolitics\\eastpolitics1000AMT\\";
    public static final String eastpoliticsSimulatedPath = basePath + "eastpolitics\\simulated1000\\";
    public static final String processedPath = basePath + "Processed\\";
    public static final String GTICPath = processedPath + "GTIC\\";
    public static final String BinaryBiasedPath = processedPath + "BinaryBiased\\";
    public static final String BinaryUnbiasedPath = processedPath + "BinaryUnbiased\\";
    public static final String income94Path = basePath + "income94crowd\\";
    public static final String leavesPath = basePath + "leaves\\";
    public static final String synthPath = "data\\synthetic\\";
    public static final String annealPath = synthPath + "anneal\\anneal.arff";
    public static final String audiologyPath = synthPath + "audiology\\audiology.arff";
    public static final String autosPath = synthPath + "autos\\autos.arff";
    public static final String balanceScalePath = synthPath + "balance-scale\\balance-scale.arff";
    public static final String biodegPath = synthPath + "biodeg\\biodeg.arff";
    public static final String breastCancerPath = synthPath + "breast-cancer\\breast-cancer.arff";
    public static final String breastWPath = synthPath + "breast-w\\breast-w.arff";
    public static final String carPath = synthPath + "car\\car.arff";
    public static final String creditAPath = synthPath + "credit-a\\credit-a.arff";
    public static final String creditGPath = synthPath + "credit-g\\credit-g.arff";
    public static final String diabetesPath = synthPath + "diabetes\\diabetes.arff";
    public static final String heartCPath = synthPath + "heart-c\\heart-c.arff";
    public static final String heartHPath = synthPath + "heart-h\\heart-h.arff";
    public static final String heartStatlogPath = synthPath + "heart-statlog\\heart-statlog.arff";
    public static final String hepatitisPath = synthPath + "hepatitis\\hepatitis.arff";
    public static final String horseColicPath = synthPath + "horse-colic\\horse-colic.arff";
    public static final String hypothyroidPath = synthPath + "hypothyroid\\hypothyroid.arff";
    public static final String ionospherePath = synthPath + "ionosphere\\ionosphere.arff";
    public static final String irisPath = synthPath + "iris\\iris.arff";
    public static final String krVsKpPath = synthPath + "kr-vs-kp\\kr-vs-kp.arff";
    public static final String laborPath = synthPath + "labor\\labor.arff";
    public static final String letterPath = synthPath + "letter\\letter.arff";
    public static final String lymphPath = synthPath + "lymph\\lymph.arff";
    public static final String mushroomPath = synthPath + "mushroom\\agaricus-lepiota.arff";
    public static final String segmentPath = synthPath + "segment\\segment.arff";
    public static final String sickPath = synthPath + "sick\\sick.arff";
    public static final String sonarPath = synthPath + "sonar\\sonar.arff";
    public static final String spambasePath = synthPath + "spambase\\spambase.arff";
    public static final String ticTacToePath = synthPath + "tic-tac-toe\\tic-tac-toe.arff";
    public static final String vehiclePath = synthPath + "vehicle\\vehicle.arff";
    public static final String votePath = synthPath + "vote\\vote.arff";
    public static final String vowelPath = synthPath + "vowel\\vowel.arff";
    public static final String waveformPath = synthPath + "waveform\\waveform.arff";
    public static final String zooPath = synthPath + "zoo\\zoo.arff";
    public static final String imageSegmentationPath = synthPath + "image_segmentation\\image_segmentation.arff";
    public static final String banknotePath = synthPath + "banknote\\banknote.arff";
    public static final String letterCondensedPath = synthPath + "letter-condensed\\letter-condensed.arff";
    public static final String vehicleBinaryPath = synthPath + "vehicle-binary\\vehicle-binary.arff";
    
    public static ArrayList<Dataset> getCrowdsourcingDatasets() throws Exception{
        ArrayList<Dataset> datasets = new ArrayList();
        
        Dataset adultDataset = FileLoader.loadFile(GTICPath + "adult2.response.txt", GTICPath + "adult2.gold.txt");
        adultDataset.setRelationName("Adult2");
        datasets.add(adultDataset);


        Dataset fejDataset = FileLoader.loadFile(GTICPath + "fej2013.response.txt", GTICPath + "fej2013.gold.txt");
        fejDataset.setRelationName("FEJ2013");
        datasets.add(fejDataset);


        Dataset duckDataset = FileLoader.loadFile(BinaryBiasedPath + "duck.response.txt", BinaryBiasedPath + "duck.gold.txt");
        duckDataset.setRelationName("Duck");
        datasets.add(duckDataset);

        Dataset incomeDataset = FileLoader.loadFile(income94Path + "income94.response.txt", income94Path + "income94.gold.txt");
        incomeDataset.setRelationName("Income94");
        datasets.add(incomeDataset);

        Dataset leavesDataset = FileLoader.loadFile(leavesPath + "leaves6.response.txt", leavesPath + "leaves6.gold.txt");
        leavesDataset.setRelationName("Leaves6");
        datasets.add(leavesDataset);

        return datasets;
    }
}
