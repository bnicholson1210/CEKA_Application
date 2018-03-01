package research.crowdsourcing;

import ceka.consensus.AdaptiveWeightedMajorityVote;
import ceka.consensus.MajorityVote;
import ceka.consensus.ds.DawidSkene;
import ceka.converters.FileLoader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;

import stat.StatCalc;
import weka.core.Matrix;
import ceka.core.Dataset;
import weka.clusterers.CLOPE;
import weka.clusterers.EM;
import ceka.core.Example;
import ceka.core.Label;
import ceka.utils.PerformanceStatistic;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import research.DatasetUtils;
import research.ResultMetrics;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.KStar;
import weka.classifiers.lazy.IBk;
import weka.clusterers.Clusterer;
import weka.clusterers.Cobweb;
import weka.clusterers.SimpleKMeans;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.IsotonicRegression;
import weka.filters.unsupervised.attribute.Normalize;

public class Analysis 
{
	public Analysis()
	{
		
	}
	
	public static void analyze(ArrayList<Dataset> datasets, ArrayList<String> names) 
			throws Exception
	{
		printDatasetsTable(datasets, names);
		File f = new File("RAnalysisCode.txt");
		f.delete();
		//First, analyze the data sets one-by-one, gaining info 
		//such as correlation among such attributes as 
		//total number of labels a worker gives vs. accuracy,
		int workerTotal = 0;
		for(int i = 0; i < datasets.size(); i++)
		{
			workerTotal += datasets.get(i).getWorkerSize();
		}
		//For this collection of all workers across all data sets considered,
		//column 0 is the proportion of all labels that the worker has contributed,
		//column 1 is the "evenness" of the worker's labels, or how equally each category is represented,
		//column 2 is the number of possible classes in the data set the worker labeled
		//column 3 is the log distance measure
		//column 4 is the estimated accuracy of the worker (by EM)
                //column 5 is the true accuracy of the worker
		double[][] allWorkerData = new double[workerTotal][6];
		int workerNum = 0;
		for(int i = 0; i < datasets.size(); i++)
		{
			Dataset dataset = datasets.get(i);
                        DawidSkene ds = new DawidSkene(30);
                        ds.doInference(dataset);
			int totalLabels = 0;
			ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
			for(int j = 0; j < workers.size(); j++)
			{
				totalLabels += workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize();
			}
			ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
			double labelEvenness = WorkerTaskGraph.getTrueLabelEvenness(tasks, dataset.getCategorySize());
			WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
			graph.initializeTendencies();
			System.out.println(names.get(i) + " label evenness: " + (new DecimalFormat("#.0000").format(labelEvenness)));
                        System.out.println(workers.size() + " workers.");
			printDatasetInfo(dataset);
                        double[] accs = new double[workers.size()];
                        for(int j = 0; j < workers.size(); j++)
                        {
                            AnalyzedWorker w = workers.get(j);
                            if(graph.getNumTasksForWorker(w) >= 9)
                                accs[j] = graph.getWorkerAccuracy(w);
                        }
			accs = StatCalc.removeZeros(accs);
                        double sd = Math.sqrt(StatCalc.variance(accs));
                        double mean = StatCalc.mean(accs);
			//For this collection only the workers in this data set are considered,
			//column 0 is the number of labels this worker has contributed,
			//column 1 is the evenness of the labels,
			//column 2 is the accuracy of the worker
			double[][] workerData = new double[workers.size()][3];
                        double[][] workerData2 = new double[workers.size()][6];
                        int localWorkerNum = 0;
                        double averageEvenness = graph.averageEvenness();
			for(int j = 0; j < workers.size(); j++)
			{
				//if(((double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
				//		/ (double)totalLabels < .001) || (double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
                                 //       < 9)
                                 //   continue;
				workerData[j][0] = workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize();
				workerData[j][1] = graph.getWorkerLabelEvenness(workers.get(j));
				workerData[j][2] = graph.getWorkerAccuracy(workers.get(j));
				
				
                                allWorkerData[workerNum][0] = 
						(double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize() /
						(double)totalLabels;
				allWorkerData[workerNum][1] = graph.getWorkerLabelEvenness(workers.get(j));
				allWorkerData[workerNum][2] = dataset.getCategorySize();
				//allWorkerData[workerNum][3] = graph.getWorkerAccuracy(workers.get(j));
                                allWorkerData[workerNum][3] = graph.getWorkerLogSimilarity(workers.get(j));
				allWorkerData[workerNum][4] = graph.getWorkerEMAccuracy(workers.get(j));
                                allWorkerData[workerNum][5] = graph.getWorkerAccuracy(workers.get(j));
                                
                                //workerData2[localWorkerNum][5] = (graph.getWorkerAccuracy(workers.get(j)) > mean + .5 * sd) ? 0 : 1;
                                workerNum++;
                                localWorkerNum++;
			}
			//System.out.println(new Matrix(StatCalc.calculateCovarianceMatrix(workerData)));
			workerData = StatCalc.removeZeros(workerData);
			createRCode(workerData, names.get(i), f);
                        workerData2 = graph.getWorkerData();
                        createIndividualSpamArff(workerData2, names.get(i), -1);
		}
		allWorkerData = StatCalc.removeZeros(allWorkerData);
		createRCode(allWorkerData, "Overall Analysis", f);
                createSpamArff(allWorkerData);
		
		//Now analyze the data sets all together, gaining info
		//such as correlation among such attributes as
		//total number of workers vs. total label quality,
		
	}
	
        //Perform clustering on dataset to see if spammers/nonspammers can easily
        //be sorted into groups.
        //WARNING: THIS FUNCTION IS DEPENDENT ON THE FUNCTION THAT CREATES THE
        //ARFF FILE! MAKE SURE THAT WHEN USING THIS FUNCTION THAT THE SAME
        //LIST OF DATASETS WAS USED TO MAKE THE ARFF FILE THAT ARE USED IN THIS
        //FUNCTION!
        public static void clusterWorkers(ArrayList<Dataset> datasets, ArrayList<String> names) throws Exception
        {
            int numClusters = 20;
            double[] clusterSims = new double[numClusters];
            MajorityVote mv = new MajorityVote();
            Dataset workers = FileLoader.loadFile("clusterSpam.arff");
            Dataset workersCopy = FileLoader.loadFile("clusterSpam.arff");
            workersCopy.setClassIndex(-1);
            Normalize normalize = new Normalize();
            normalize.setInputFormat(workersCopy);
            normalize.useFilter(workersCopy, normalize);
            for(int i = 0; i < workersCopy.getExampleSize(); i++)
            {
                Example e = workersCopy.getExampleByIndex(i);
                e.setValue(3, 0);
                //e.setValue(2, 0);
                //e.setValue(2, e.value(2)*3.0);
            }
            SimpleKMeans c = new SimpleKMeans();
            //c.setNumClusters(numClusters);
            //EM c = new EM();
            c.setNumClusters(numClusters);
            c.buildClusterer(workersCopy);
            numClusters = c.numberOfClusters();
            ArrayList<ArrayList<Integer>> clusterGroups = new ArrayList();
            for(int i = 0; i < numClusters; i++)
            {
                clusterGroups.add(new ArrayList());
            }
            for(int i = 0; i < workersCopy.getExampleSize(); i++)
            {
                int cluster = c.clusterInstance(workersCopy.getExampleByIndex(i));
                clusterGroups.get(cluster).add(i);
            }
            for(int i = 0; i < numClusters; i++)
            {
                System.out.println("Cluster " + (i + 1) + ":");
                double sum = 0;
                double evenSum = 0;
                double labelSum = 0;
                double simSum = 0;
                for(int j = 0; j < clusterGroups.get(i).size(); j++)
                {
                    Example e = workers.getExampleByIndex(clusterGroups.get(i).get(j));
                    double acc = e.value(3);
                    sum += acc;
                    evenSum += e.value(0);
                    labelSum += e.value(1);
                    simSum += e.value(2);
                    
                    //System.out.println("\t" + acc);
                    //System.out.println(workers.getExampleByIndex(clusterGroups.get(i).get(j)));
                }
                double avg = sum / (double)clusterGroups.get(i).size();
                double num = (double)clusterGroups.get(i).size();
                System.out.println("Average accuracy of workers in Cluster " + (i + 1) + ": " + avg);
                System.out.println("Number of workers in Cluster " + (i + 1) + ": " + clusterGroups.get(i).size());
                System.out.println("Average sq. dif. evenness of workers in Cluster " + (i + 1) + ": " + evenSum / num);
                System.out.println("Average prop. of tasks done of workers in Cluster " + (i + 1) + ": " + labelSum / num);
                System.out.println("Average log sim measure of workers in Cluster " + (i + 1) + ": " + simSum / num);
                clusterSims[i] = simSum / num;
            }
            
            int clusterToRemove = 3;
            for(int i = 0; i < workersCopy.getExampleSize(); i++)
            {
                Example e = workersCopy.getExampleByIndex(i);
                for(int j = 0; j < clusterGroups.get(clusterToRemove).size(); j++)
                {
                    if(i == clusterGroups.get(clusterToRemove).get(j))
                    {
                        e.setValue(4, 1);
                        e.setTrueLabel(new Label("", "1", "", ""));
                    }
                }
                //System.out.println(e);
            }
            workersCopy.randomize(new Random());
            workersCopy.setClassIndex(4);
            System.out.println("Validation accuracy: " + ResultMetrics.validation(workersCopy, new J48(), 10));
            
            int counter = 0;
            for(int i = 0; i < datasets.size(); i++)
            {
                Dataset dataset = datasets.get(i);
                mv.doInference(dataset);
                System.out.println("Accuracy of " + names.get(i) + ": " + ResultMetrics.accuracy(dataset));
                ArrayList<AnalyzedWorker> spammers = new ArrayList();
                ArrayList<AnalyzedWorker> possibleSpammers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
                for(int j = 0; j < possibleSpammers.size(); j++)
                {
                    for(int k = 0; k < clusterGroups.get(clusterToRemove).size(); k++)
                    {
                        if(counter == clusterGroups.get(clusterToRemove).get(k))
                        {
                            spammers.add(possibleSpammers.get(j));
                            break;
                        }
                    }
                    counter++;
                }
                ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
                WorkerTaskGraph graph = new WorkerTaskGraph(dataset, possibleSpammers, tasks);
                System.out.println("Label Quality of " + names.get(i) + ": " + graph.getLabelQuality());
                Dataset newDataset = graph.removeSpammers(spammers);
                mv.doInference(newDataset);
                ArrayList<AnalyzedWorker> workers2 = SyntheticCrowdsourcing.getWorkersForDataset(newDataset);
                ArrayList<AnalyzedTask> tasks2 = SyntheticCrowdsourcing.getTasksForDataset(newDataset);
                WorkerTaskGraph graph2 = new WorkerTaskGraph(newDataset, workers2, tasks2);
                System.out.println("Now accuracy of " + names.get(i) + ": " + ResultMetrics.accuracy(newDataset));
                System.out.println("Now Label Quality of " + names.get(i) + ": " + graph2.getLabelQuality() + "\n");
            }
        }
        
        
	//Generates r code to examine the covariance of the data.
	public static void createRCode(double[][] data, String message, File f) throws Exception
	{
		BufferedWriter bw = new BufferedWriter(new FileWriter(f, true));
		bw.write("# " + message + "\n");
		DecimalFormat df = new DecimalFormat("#.#####");
		String[] x = new String[data[0].length];
		for(int i = 0; i < x.length; i++)
		{
			x[i] = "x" + (i + 1) + " = c(";
		}
		for(int i = 0; i < data.length; i++)
		{
			for(int j = 0; j < data[i].length; j++)
			{
				x[j] += "" + df.format(data[i][j]);
				if(i != data.length - 1)
					x[j] += ",";
				if(i % 60 == 0 && i > 0)
					x[j] += "\n";
			}
		}
		for(int i = 0; i < x.length; i++)
		{
			x[i] += ")\n";
			bw.write(x[i]);
		}
		bw.write("X = cbind(");
		for(int i = 0; i < x.length; i++)
		{
			bw.write("x" + (i + 1));
			if(i != x.length - 1)
				bw.write(", ");
		}
		bw.write(")\n");
		String[] colnames = new String[0];
		if(data[0].length == 3)
		{
			colnames = new String[3];
			colnames[0] = "#Labels";
			colnames[1] = "Evenness";
			colnames[2] = "Accuracy";
			bw.write("colnames(X) = c(\"" + colnames[0] + "\",\"" + colnames[1] + "\",\"" + colnames[2] + "\")\n");
		}
		else if(data[0].length == 5)
		{
			colnames = new String[5];
			colnames[0] = "%Labels";
			colnames[1] = "Worker Evenness";
			colnames[2] = "#Classes";
			colnames[3] = "Accuracy";
			colnames[4] = "Data Set Evenness";
			bw.write("colnames(X) = c(\"" + colnames[0] + "\",\"" + colnames[1] + 
					"\",\"" + colnames[2] + "\",\"" + colnames[3] + "\", \"" + colnames[4] +"\")\n");
		}
		bw.write("# " + message + "\n");
		bw.write("rcorr(X)\n\n");
		bw.write(scatterplots(colnames, message));
		bw.close();
	}
	
	public static void printDatasetInfo(Dataset dataset)
	{
		System.out.println("Num classes " + dataset.getCategorySize());
		int[] counters = new int[dataset.getCategorySize()];
		for(int i = 0; i < dataset.getExampleSize(); i++)
		{
			counters[dataset.getExampleByIndex(i).getTrueLabel().getValue()]++;
		}
		for(int i = 0; i < counters.length; i++)
		{
			System.out.print(counters[i] + " ");
		}
		System.out.println();
	}
	
	public static void printDatasetsTable(ArrayList<Dataset> datasets, ArrayList<String> names)
	{
		DecimalFormat df = new DecimalFormat("0.0000");
		for(int i = 0; i < datasets.size(); i++)
		{
			Dataset dataset = datasets.get(i);
			ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
			double labelEvenness = WorkerTaskGraph.getTrueLabelEvenness(tasks, dataset.getCategorySize());
			ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
			int totalLabels = 0;
			for(int j = 0; j < workers.size(); j++)
			{
                            totalLabels += workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize();
			}
			System.out.print(names.get(i) + " & " + tasks.size() + " & " + totalLabels
			 + " & " + dataset.getCategorySize() + " & " + df.format(labelEvenness) + "\\\\\n\\hline\n");
		}
	}
	
	public static String scatterplots(String[] varNames, String datasetName)
	{
		int length = varNames.length;
		String result = "";
		for(int i = 0; i < length; i++)
		{
			for(int j = i + 1; j < length; j++)
			{
				String qual1 = varNames[i].replace("%","Percentage").replace(" ", "");
				String qual2 = varNames[j].replace("%","Percentage").replace(" ", "");

				result += "png('C:\\\\Users\\\\Bryce\\\\Desktop\\\\ScatterPlots\\\\" + 
						qual1 + "VS" + qual2 + "." + datasetName + ".png')\n";
				result += "plot(x" + (i + 1) + ",x" + (j + 1) + ", xlab = \"" + varNames[i]
						+ "\", ylab = \"" + varNames[j] + "\", main = \"" + datasetName + " Correlation Plot\""
								+ ", cex.main = 2.1, cex.lab = 1.5, cex.axis = 1.7)\n";
				result += "dev.off()\n";
			}
		}
		return result;
	}
	
	//Creates a .arff file that contains each worker's evenness, #labels, and 
	//Similarity measure, and class value of whether or not the worker is a spammer
	public static void createSpammerArff(Dataset dataset, String name) throws Exception
	{
		ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
		ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
		WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
		double[][] data = new double[workers.size()][4];
		for(int i = 0; i < workers.size(); i++)
		{
			AnalyzedWorker worker = workers.get(i);
			data[i][0] = graph.getWorkerLabelEvenness(worker);
			data[i][1] = worker.getMultipleNoisyLabelSet(0).getLabelSetSize();
			data[i][2] = graph.getWorkerSimilarity(worker);
			if(graph.getWorkerAccuracy(worker) >= (1.2 / (double)dataset.getCategorySize()))
				data[i][3] = 0;
			else
				data[i][3] = 1;
		}
		
		File f = new File("SpammerArffs\\" + name + ".arff");
		f.delete();
		BufferedWriter bw = new BufferedWriter(new FileWriter(f));
		bw.write("@relation\t" + name + "\n");
		bw.write("@attribute\tatt1\treal\n");
		bw.write("@attribute\tatt2\treal\n");
		bw.write("@attribute\tatt3\treal\n");
		bw.write("@attribute\tclass\t{0,1}\n");
                
		bw.write("@data\n");
		
                writeInstances(data, bw);
		bw.close();
	}
        
        public static void createTotalSpammerArff(ArrayList<Dataset> datasets, ArrayList<String> names) throws Exception
        {
            for(int n = 0; n < datasets.size(); n++)
            {
                Dataset dataset = datasets.get(n);
                String name = names.get(n);
                System.out.println("For dataset " + names.get(n));
                ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
                ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
                WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
                double datasetEvenness = graph.getTrueLabelEvenness(tasks, dataset.getCategorySize());
                double[][] data = new double[workers.size()][4];
                for(int i = 0; i < workers.size(); i++)
                {
                    AnalyzedWorker worker = workers.get(i);
                    data[i][0] = graph.getWorkerLabelEvenness(worker);
                    data[i][1] = worker.getMultipleNoisyLabelSet(0).getLabelSetSize();
                    data[i][2] = graph.getWorkerSimilarity(worker);
                    System.out.println("\tWorker " + (i + 1));
                    System.out.println("\t\tWorker Label Evenness: " + data[i][0]);
                    System.out.println("\t\tDataset Evenness: " + datasetEvenness);
                    System.out.println("\t\tWorker Relative Evenness: " + graph.getWorkerRelativeEvenness(worker));
                    System.out.println("\t\tWorker Similarity Measure: " + data[i][2]);
                    System.out.println("\t\tNumber of Labels: " + data[i][1]);
                    System.out.println("\t\tAccuracy: " + graph.getWorkerAccuracy(worker));
                    
                    if(graph.getWorkerAccuracy(worker) >= (1.2 / (double)dataset.getCategorySize()))
                    {
                            data[i][3] = 0;
                            System.out.println("\tI don't this worker is a spammer.");
                    }
                    else
                    {
                            data[i][3] = 1;
                            System.out.println("\tI think this worker is a spammer.");
                    }
                }
                /*
                File f = new File("SpammerArffs\\" + name + ".arff");
                f.delete();
                BufferedWriter bw = new BufferedWriter(new FileWriter(f));
                bw.write("@relation\t" + name + "\n");
                bw.write("@attribute\tatt1\treal\n");
                bw.write("@attribute\tatt2\treal\n");
                bw.write("@attribute\tatt3\treal\n");
                bw.write("@attribute\tclass\t{");
                for(int i = 0; i < dataset.getCategorySize(); i++)
                {
                    bw.write("" + i);
                    if(i != dataset.getCategorySize() - 1)
                        bw.write(",");
                }
                bw.write("}\n");

                bw.write("@data\n");

                for(int i = 0; i < data.length; i++)
                {
                    if(data[i][1] >= 10)
                    {
                        bw.write(data[i][0] + ",");
                        bw.write(data[i][1] + ",");
                        bw.write(data[i][2] + ",");
                        bw.write((int)data[i][3] + "\n");
                    }
                }
                bw.close();
                        */
            }
        }
        
        public static void writeInstances(double[][] data, BufferedWriter bw) throws Exception
        {
            for(int i = 0; i < data.length; i++)
		{
                    if(data[i][1] >= 10)
                    {
			bw.write(data[i][0] + ",");
			bw.write(data[i][1] + ",");
			bw.write(data[i][2] + ",");
			bw.write((int)data[i][3] + "\n");
                    }
                }
        }
        
        //column 0 is the proportion of all labels that the worker has contributed,
        //column 1 is the "evenness" of the worker's labels, or how equally each category is represented,
        //*column 2 is the number of possible classes in the data set the worker labeled
        //column 3 is the log distance measure
        //column 4 is the estimated accuracy of the worker (by EM)
        //*column 5 is the true accuracy of the worker
        //column 6 is the spammer score (RY) of the worker
        //column 7 is the worker cost (IPW) of the worker
        //column 8 is the average relative delta entropy of the worker
        //column 9 is the average cosine similarity with all other applicable workers
        //*currently not used attributes
        public static void createClusterArff(ArrayList<Dataset> datasets) throws Exception
        {
            File f = new File("clusterSpam.arff");
            f.delete();
            DecimalFormat df = new DecimalFormat("#.####");
            BufferedWriter bw = new BufferedWriter(new FileWriter(f));
            bw.write("@RELATION\tclusterSpam\n");
            bw.write("@ATTRIBUTE\tatt1\treal\n");
            bw.write("@ATTRIBUTE\tatt2\treal\n");
            //bw.write("@ATTRIBUTE\tatt3\treal\n");
            bw.write("@ATTRIBUTE\tatt4\treal\n");
            bw.write("@ATTRIBUTE\tatt5\treal\n");
            //bw.write("@ATTRIBUTE\tatt6\treal\n");
            bw.write("@ATTRIBUTE\tatt7\treal\n");
            bw.write("@ATTRIBUTE\tatt8\treal\n");
            bw.write("@ATTRIBUTE\tatt9\treal\n");
            bw.write("@ATTRIBUTE\tatt10\treal\n");
            bw.write("@ATTRIBUTE\tclass\t{0,1}\n");
            //bw.write("@ATTRIBUTE\tclass\treal\n");
            bw.write("\n@DATA\n");
            for(int i = 0; i < datasets.size(); i++)
            {
                Dataset dataset = datasets.get(i);
                new DawidSkene(30).doInference(dataset);
                ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
                ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
                WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
                //First attribute is square difference from average evenness, 
                //second is percentage of total labels contributed, third is 
                //label similarity measure, fourth is accuracy
                double averageEvenness = graph.averageEvenness();
                for(int j = 0; j < workers.size(); j++)
                {
                    AnalyzedWorker w = workers.get(j);
                    double prop = (double)(w.getMultipleNoisyLabelSet(0).getLabelSetSize()) /
                            (double)graph.getTotalNumLabels();
                    bw.write(df.format(prop) + ",");
                    //System.out.println("BLARG\t" + Math.abs(graph.getWorkerLabelEvenness(w) - averageEvenness));
                    bw.write(df.format(Math.abs(graph.getWorkerLabelEvenness(w) - averageEvenness)) + ",");
                    //bw.write(df.format(graph.getWorkerLabelEvenness(w)) + ",");
                    //bw.write(df.format(dataset.getCategorySize()) + ",");
                    //bw.write(df.format(graph.getWorkerSimilarity(w)) + ",");
                    bw.write(df.format(graph.getWorkerLogSimilarity(w)) + ",");
                    bw.write(df.format(graph.getWorkerEMAccuracy(w)) + ",");
                    //bw.write(df.format(graph.getWorkerAccuracy(w)) + ",");
                    //bw.write(df.format(graph.getWorkerAccuracy(w)) + "\n");
                    
                    bw.write(df.format(graph.spammerScore(w, null)[0]) + ",");
                    //bw.write("" + 0 + ",");
                    bw.write(df.format(graph.workerCost(w, null)[0]) + ",");
                    bw.write(df.format(graph.getWorkerRelativeDeltaEntropy(w)) + ",");
                    bw.write(df.format(graph.getAverageSimilarityWithAllOtherWorkers(w)) + ",");
                    //bw.write("" + 0 + ",");
                    bw.write("0\n");
                }
            }
            bw.close();
        }
        
        public static String createWorkerAttributesArff(ArrayList<Dataset> datasets, ArrayList<String> attributeNames)
            throws Exception{
            String filename = "workerAttributes.arff";
             File f = new File(filename);
            f.delete();
            DecimalFormat df = new DecimalFormat("#.####");
            BufferedWriter bw = new BufferedWriter(new FileWriter(f));
            bw.write("@RELATION\tworkerAttributes\n");
            for(int i = 0; i < attributeNames.size(); i++){
                bw.write("@ATTRIBUTE\tatt" + (i + 1) + "\treal\n");
            }
            bw.write("@ATTRIBUTE\tclass\t{0,1}\n");
            bw.write("\n@DATA\n");
            for(int i = 0; i < datasets.size(); i++)
            {
                Dataset dataset = datasets.get(i);
                new DawidSkene(30).doInference(dataset);
                ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
                ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
                WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
                for(int j = 0; j < workers.size(); j++)
                {
                    AnalyzedWorker w = workers.get(j);
                    for(int k = 0; k < attributeNames.size(); k++){
                        bw.write(df.format(graph.getCharacteristicValueForWorker(attributeNames.get(k), w)) + ",");
                    }
                    bw.write("0\n");
                }
            }
            bw.close();
            return filename;
        }
        
        public static String createWorkerAttributesArffWithSpammerIndicatorLabel(Dataset dataset, 
                String name, ArrayList<String> attributeNames, String evalAttribute, double prop)
                throws Exception {
            File spammerArffsDirectory = new File("SpammerArffs");
            if(!spammerArffsDirectory.exists()){
                spammerArffsDirectory.mkdir();
            }
            File thisSpammerArffDirectory = new File("SpammerArffs\\" + name);
            if(!thisSpammerArffDirectory.exists()){
                thisSpammerArffDirectory.mkdir();
            }
            String filename = "SpammerArffs\\" + name + "\\" + name + ".arff";
            File f = new File(filename);
            f.delete();
            DecimalFormat df = new DecimalFormat("#.####");
            BufferedWriter bw = new BufferedWriter(new FileWriter(f));
            bw.write("@RELATION\t" + name + "Spammers\n");
            for(int i = 0; i < attributeNames.size(); i++){
                bw.write("@ATTRIBUTE\tatt" + (i + 1) + "\treal\n");
            }
            bw.write("@ATTRIBUTE\tclass\t{0,1}\n");
            bw.write("\n@DATA\n");
            new DawidSkene(30).doInference(dataset);
            ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
            ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
            
            ArrayList<Double> evalAttValues = new ArrayList();
            for(int i = 0; i < workers.size(); i++){
                evalAttValues.add(graph.getCharacteristicValueForWorker(evalAttribute, workers.get(i)));
            }
            Collections.sort(evalAttValues);
            double threshVal = evalAttValues.get((int)(prop * evalAttValues.size()));
            for(int j = 0; j < workers.size(); j++)
            {
                AnalyzedWorker w = workers.get(j);
                for(int k = 0; k < attributeNames.size(); k++){
                    bw.write(df.format(graph.getCharacteristicValueForWorker(attributeNames.get(k), w)) + ",");
                }
                if(graph.getCharacteristicValueForWorker(evalAttribute, w) > threshVal)
                    bw.write("0\n");
                else
                    bw.write("1\n");
            }
            bw.close();
            return filename;
        }
       
        //column 0 is the proportion of all labels that the worker has contributed,
        //column 1 is the "evenness" of the worker's labels, or how equally each category is represented,
        //*column 2 is the number of possible classes in the data set the worker labeled
        //column 3 is the log distance measure
        //column 4 is the estimated AUC of the worker (by EM)
        //*column 5 is the true accuracy of the worker
        //column 6 is the spammer score (RY) of the worker
        //column 7 is the worker cost (IPW) of the worker
        //*currently not used attributes
        public static void createClusterArffAUC(ArrayList<Dataset> datasets) throws Exception
        {
            File f = new File("clusterSpam.arff");
            f.delete();
            DecimalFormat df = new DecimalFormat("#.####");
            BufferedWriter bw = new BufferedWriter(new FileWriter(f));
            bw.write("@RELATION\tclusterSpam\n");
            bw.write("@ATTRIBUTE\tatt1\treal\n");
            bw.write("@ATTRIBUTE\tatt2\treal\n");
            //bw.write("@ATTRIBUTE\tatt3\treal\n");
            bw.write("@ATTRIBUTE\tatt4\treal\n");
            bw.write("@ATTRIBUTE\tatt5\treal\n");
            //bw.write("@ATTRIBUTE\tatt6\treal\n");
            bw.write("@ATTRIBUTE\tatt7\treal\n");
            bw.write("@ATTRIBUTE\tatt8\treal\n");
            bw.write("@ATTRIBUTE\tclass\t{0,1}\n");
            //bw.write("@ATTRIBUTE\tclass\treal\n");
            bw.write("\n@DATA\n");
            for(int i = 0; i < datasets.size(); i++)
            {
                Dataset dataset = datasets.get(i);
                new DawidSkene(30).doInference(dataset);
                ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
                ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
                WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);

                double averageEvenness = graph.averageEvenness();
                for(int j = 0; j < workers.size(); j++)
                {
                    AnalyzedWorker w = workers.get(j);
                    double prop = (double)(w.getMultipleNoisyLabelSet(0).getLabelSetSize()) /
                            (double)graph.getTotalNumLabels();
                    bw.write(df.format(prop) + ",");
                    bw.write(df.format(Math.abs(graph.getWorkerLabelEvenness(w) - averageEvenness)) + ",");
                    //bw.write(df.format(graph.getWorkerLabelEvenness(w)) + ",");
                    //bw.write(df.format(dataset.getCategorySize()) + ",");
                    //bw.write(df.format(graph.getWorkerSimilarity(w)) + ",");
                    bw.write(df.format(graph.getWorkerLogSimilarity(w)) + ",");
                    bw.write(df.format(graph.getWorkerEMAUC(w)) + ",");
                    //bw.write(df.format(graph.getWorkerAccuracy(w)) + ",");
                    //bw.write(df.format(graph.getWorkerAccuracy(w)) + "\n");
                    
                    bw.write(df.format(graph.spammerScore(w, null)[0]) + ",");
                    //bw.write("" + 0 + ",");
                    bw.write(df.format(graph.workerCost(w, null)[0]) + ",");
                    //bw.write("" + 0 + ",");
                    bw.write("0\n");
                }
            }
            bw.close();
        }
        
        public static double[] filterSpammersJ48(Dataset dataset, String name, ArrayList<String> names) throws Exception
        {
            ArrayList<AnalyzedWorker> spammers = new ArrayList();
            ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
            ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
            double totalLabels = 0;
            for(int j = 0; j < workers.size(); j++)
            {
                totalLabels += workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize();
            }
            System.out.println(name);
           
            //DawidSkene ds = new DawidSkene(30);
            DawidSkene ds = new DawidSkene(30);
            ds.doInference(dataset);
            Dataset spamDataset = FileLoader.loadFile("SpammerArffs\\" + name + "\\" + name + ".arff");
            Dataset thisDataset = FileLoader.loadFile("SpammerArffs\\" + name + "\\" + name + ".arff");
            spamDataset = spamDataset.generateEmpty();
            for(int i = 0; i < names.size(); i++)
            {
                if(!name.equals(names.get(i)))
                {
                    spamDataset = DatasetUtils.combineDatasets(spamDataset, FileLoader.loadFile("SpammerArffs\\" + names.get(i) + "\\" + names.get(i) + ".arff"));
                }
            }
            J48 j48 = new J48();
            j48.buildClassifier(spamDataset);
            for(int i = 0; i < thisDataset.getExampleSize(); i++)
            {
                if(j48.classifyInstance(thisDataset.getExampleByIndex(i)) == 1)
                {
                    System.out.println("We think that worker " + i + " is a spammer!");
                    spammers.add(workers.get(i));
                    
                }
            }
           
            
            
            double acc1 = ResultMetrics.accuracy(dataset);
            Dataset filteredDataset = graph.removeSpammers(spammers);
            ds = new DawidSkene(30);
            ds.doInference(filteredDataset);
            double acc2 = ResultMetrics.accuracy(filteredDataset);
            System.out.println("\tAccuracy of dataset: " + acc1);
            System.out.println("\tAccuracy of filtered dataset: " + acc2);
            System.out.println("\tPercent increase: " + (acc2 - acc1));
            double[] result = new double[2];
            result[0] = acc1;
            result[1] = acc2;
            return result;
        }
        /*
        public static void filterSpammers(Dataset dataset, String name, double threshold)
        {
            System.out.println("Filtering spammers for dataset " + name);
            boolean go;
            do
            {
                go = false;
                MajorityVote ds = new MajorityVote();
                //DawidSkene ds = new DawidSkene(30);
                ds.doInference(dataset);
                System.out.println(MainProc.accuracy(dataset));
                ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
                ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
                WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
                double[] logSims = new double[workers.size()];
                for(int i = 0; i < workers.size(); i++)
                {
                    logSims[i] = graph.getWorkerLogSimilarity(workers.get(i));
                }
                double sd = Math.sqrt(StatCalc.variance(logSims));
                double mean = StatCalc.mean(logSims);
                double max = Double.NEGATIVE_INFINITY;
                int maxIndex = -1;
                for(int i = 0; i < workers.size(); i++)
                {
                    if(logSims[i] > max)
                    {
                        max = logSims[i];
                        maxIndex = i;
                    }
                }
                if(max >= mean + sd * threshold)
                {
                    go = true;
                    ArrayList<AnalyzedWorker> sp = new ArrayList();
                    sp.add(workers.get(maxIndex));
                    dataset = graph.removeSpammers(sp);
                }
            }while(go);
        }
        */
        
        //New method (that doesn't work...)
        /*
        public static void filterSpammers(Dataset dataset, String name)
        {
            System.out.println("For dataset " + name);
            int numGuesses = 1000;
            int maxIterations = 10;
            Random rand = new Random();
            ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
            ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
            double[] sims = new double[workers.size()];
            for(int i = 0; i < workers.size(); i++)
            {
                sims[i] = graph.getWorkerLogSimilarity(workers.get(i));
            }
            double[] maxAndMinSims = graph.maxAndMinLogSims();
            double[] accApproxs = new double[workers.size()];
            //First find the average accuracy of the data set
            //using a least-squares approach
            for(int iteration = 0; iteration < maxIterations; iteration++)
            {
                System.out.println(iteration);
                double maxGuess = Double.NEGATIVE_INFINITY;
                double minSum = Double.POSITIVE_INFINITY;
                for(int guess = 0; guess < numGuesses; guess++)
                {
                    double g = rand.nextDouble();
                    double squaresSum = 0;
                    //double logSum = 0;
                    for(int i = 0; i < workers.size(); i++)
                    {
                        AnalyzedWorker w = workers.get(i);
                        double numTasks = graph.allTasksForWorker(w).size();
                        double sim = sims[i];
                        if(iteration == 0)
                        {
                            double acc = 1.0 - ((Math.log(sim) - Math.log(maxAndMinSims[1])) / (Math.log(maxAndMinSims[0]) - Math.log(maxAndMinSims[1])));
                            accApproxs[i] = acc;
                        }
                        double approx = -1 * (accApproxs[i] * numTasks * Math.log(g) + (1.0 - accApproxs[i]) * numTasks * Math.log(1.0 - g));
                        squaresSum += numTasks * Math.sqrt(Math.pow(approx - sim, 2.0));
                        //double diff = approx - sim;
                        //if(diff != 0)
                        //{
                        //    logSum += numTasks * Math.log(Math.abs(approx - sim));
                        //}
                    }
                    //System.out.println("A guess of " + g + " yields a total error of " + squaresSum);
                    if(squaresSum < minSum)
                    {
                        minSum = squaresSum;
                        maxGuess = g;
                    }
                }

                double averageAccuracyApproximation = maxGuess;
                //Now start approximating each worker's accuracy
                for(int i = 0; i < workers.size(); i++)
                {
                    AnalyzedWorker w = workers.get(i);
                    double sim = sims[i];
                    double avg = averageAccuracyApproximation;
                    double numTasks = graph.allTasksForWorker(w).size();
                    double numerator = -1 * numTasks * sim - numTasks * Math.log(1 - avg);
                    double denominator = numTasks * Math.log(avg) - numTasks * Math.log(1 - avg);
                    accApproxs[i] = numerator / denominator;
                    System.out.println("Sim is " + sim + ", calculated average is " + avg + ", calculated accuracy is " + accApproxs[i] + ", real accuracy is " + graph.getWorkerAccuracy(w));
                }
            }
            for(int i = 0; i < workers.size(); i++)
            {
                System.out.println("\t" + i + "\t" + accApproxs[i] + "\t" + graph.getWorkerAccuracy(workers.get(i)));
            }
        }
        */
        public static void printCorrelationCode(Dataset dataset, String name)
        {
            String props = "prop = c(";
            String even = "dae = c(";
            String log = "log = c(";
            String acc = "acc = c(";
            String emacc = "emacc = c(";
            new DawidSkene(30).doInference(dataset);
            ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
            ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
            DecimalFormat df = new DecimalFormat("#.####");
            double averageEvenness = graph.averageEvenness();
            double count = 1;
            for(int i = 0; i < workers.size(); i++)
            {
                AnalyzedWorker w = workers.get(i);
                double prop = (double)(w.getMultipleNoisyLabelSet(0).getLabelSetSize()) /
                        (double)graph.getTotalNumLabels();
                even += (df.format(Math.abs(graph.getWorkerLabelEvenness(w) - averageEvenness)));
                //bw.write(df.format(graph.getWorkerRelativeEvenness(w)) + ",");
                props += (df.format(prop));
                //bw.write(df.format(graph.getWorkerSimilarity(w)) + ",");
                log += df.format(graph.getWorkerLogSimilarity(w));
                acc += df.format(graph.getWorkerAccuracy(w));
                emacc += df.format(graph.getWorkerEMAccuracy(w));
                if(i != workers.size() - 1)
                {
                    even += ",";
                    props += ",";
                    log += ",";
                    acc += ",";
                    emacc += ",";
                }
                
                if(count % 30 == 0)
                {
                    even += "\n";
                    props += "\n";
                    log += "\n";
                    acc += "\n";
                    emacc += ",";
                }
            }
            System.out.println(even + ")\n");
            System.out.println(props + ")\n");
            System.out.println(log + ")\n");
            System.out.println(acc + ")\n");
            System.out.println(emacc += ")\n");
            System.out.println("X = cbind(dae, prop, log, emacc, acc)\n");
            System.out.println("#For dataset " + name);
            System.out.print("cor(X)\n");
        }
        
        public static void printCorrelationCodeAUC(Dataset dataset, String name)
        {
            String props = "prop = c(";
            String even = "dae = c(";
            String log = "log = c(";
            String auc = "auc = c(";
            String emauc = "emauc = c(";
            new DawidSkene(30).doInference(dataset);
            ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
            ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
            DecimalFormat df = new DecimalFormat("#.####");
            double averageEvenness = graph.averageEvenness();
            double count = 1;
            for(int i = 0; i < workers.size(); i++)
            {
                AnalyzedWorker w = workers.get(i);
                double prop = (double)(w.getMultipleNoisyLabelSet(0).getLabelSetSize()) /
                        (double)graph.getTotalNumLabels();
                even += (df.format(Math.abs(graph.getWorkerLabelEvenness(w) - averageEvenness)));
                //bw.write(df.format(graph.getWorkerRelativeEvenness(w)) + ",");
                props += (df.format(prop));
                //bw.write(df.format(graph.getWorkerSimilarity(w)) + ",");
                log += df.format(graph.getWorkerLogSimilarity(w));
                auc += df.format(graph.getWorkerAUC(w));
                emauc += df.format(graph.getWorkerEMAUC(w));
                if(i != workers.size() - 1)
                {
                    even += ",";
                    props += ",";
                    log += ",";
                    auc += ",";
                    emauc += ",";
                }
                
                if(count % 30 == 0)
                {
                    even += "\n";
                    props += "\n";
                    log += "\n";
                    auc += "\n";
                    emauc += ",";
                }
            }
            System.out.println(even + ")\n");
            System.out.println(props + ")\n");
            System.out.println(log + ")\n");
            System.out.println(auc + ")\n");
            System.out.println(emauc += ")\n");
            System.out.println("X = cbind(dae, prop, log, emauc, auc)\n");
            System.out.println("#For dataset " + name);
            System.out.print("cor(X)\n");
        }
        
        //For this collection of all workers across all data sets considered,
		//column 0 is the proportion of all labels that the worker has contributed,
		//column 1 is the "evenness" of the worker's labels, or how equally each category is represented,
		//column 2 is the number of possible classes in the data set the worker labeled
		//column 3 is the log distance measure
		//column 4 is the estimated accuracy of the worker (by EM)
                //column 5 is the true accuracy of the worker
        public static void createSpamArff(double[][] data) throws Exception
        {
            DecimalFormat df = new DecimalFormat("#.#####");
            File f = new File("workerSpam.arff");
            f.delete();
            BufferedWriter bw = new BufferedWriter(new FileWriter(f));
            bw.write("@RELATION Spam\n\n");
            bw.write("@ATTRIBUTE proportion NUMERIC\n");
            bw.write("@ATTRIBUTE evenness NUMERIC\n");
            bw.write("@ATTRIBUTE numclasses NUMERIC\n");
            bw.write("@ATTRIBUTE logsim NUMERIC\n");
            bw.write("@ATTRIBUTE emacc NUMERIC\n");
            bw.write("@ATTRIBUTE class {0,1}\n\n");
            bw.write("@DATA\n\n");
            for(int i = 0; i < data.length; i++)
            {
                for(int j = 0; j < data[i].length - 1; j++)
                {
                    bw.write(df.format(data[i][j]) + ",");
                }
                if(data[i][5] <= .60)
                    bw.write("1\n");
                else
                    bw.write("0\n");
            }
            bw.close();
        }
        
        //column 0 is the proportion of all labels that the worker has contributed,
        //column 1 is the "evenness" of the worker's labels, or how equally each category is represented,
        //*column 2 is the number of possible classes in the data set the worker labeled
        //column 3 is the log distance measure
        //*column 4 is the estimated accuracy of the worker (by EM)
        //*column 5 is the true accuracy of the worker
        //column 6 is the spammer score (RY) of the worker
        //column 7 is the worker cost (IPW) of the worker
        //*currently not used attributes
        public static void createIndividualSpamArff(double[][] data, String name, double proportion) throws Exception
        {
            DecimalFormat df = new DecimalFormat("#.#####");
            File f = new File("SpammerArffs\\" + name + "\\" + name + ".arff");
            f.delete();
            BufferedWriter bw = new BufferedWriter(new FileWriter(f));
            bw.write("@RELATION " + name + "Spam\n\n");
            bw.write("@ATTRIBUTE proportion NUMERIC\n");
            bw.write("@ATTRIBUTE evenness NUMERIC\n");
            //bw.write("@ATTRIBUTE numclasses NUMERIC\n");
            bw.write("@ATTRIBUTE logsim NUMERIC\n");
            bw.write("@ATTRIBUTE emacc NUMERIC\n");
            bw.write("@ATTRIBUTE spamscore NUMERIC\n");
            bw.write("@ATTRIBUTE cost NUMERIC\n");
            bw.write("@ATTRIBUTE class {0,1}\n\n");
            //bw.write("@ATTRIBUTE class NUMERIC\n\n");
            bw.write("@DATA\n\n");
            double prop = 0;
            double threshold = 0;
            double[] nums = new double[data.length];
            for(int i = 0; i < data.length; i++)
            {
                nums[i] = data[i][4];
            }
            if(proportion == -1)
            {
                prop = .80;
                threshold = .80;
            }
            else
            {
                //threshold = StatCalc.findPercentile(nums, proportion);
                threshold = proportion;
            }
            for(int i = 0; i < data.length; i++)
            {
                boolean record = false;
                for(int j = 0; j < data[i].length; j++)
                {
                    if(data[i][j] == 0)
                        continue;
                    else
                    {
                        record = true;
                        break;
                    }
                }
                if(record)
                {
                    for(int j = 0; j < data[i].length; j++)
                    {
                        if(j != 2 && j != 4 && !(j == 5 || j == 6))
                        
                        {
                            //System.out.println("Blarg2\t" + data[i][j]);
                            bw.write(df.format(data[i][j]) + ",");
                        }
                        if(j == 5 || j == 6)
                            //bw.write("" + 0 + ",");
                             bw.write(df.format(data[i][j]) + ",");
                        if(j == 4)
                            bw.write("" + 0 + ",");
                    }
                    
                    
                    if(data[i][4] > threshold)
                    //if(data[i][5] == 0)
                        bw.write("0\n");
                    else
                        bw.write("1\n");
                    
                    //For regression
                    //bw.write(df.format(data[i][5]) + "\n");
                }
            }
            bw.close();
        }
        
        //column 0 is the proportion of all labels that the worker has contributed,
        //column 1 is the "evenness" of the worker's labels, or how equally each category is represented,
        //*column 2 is the number of possible classes in the data set the worker labeled
        //column 3 is the log distance measure
        //*column 4 is the estimated accuracy of the worker (by EM)
        //*column 5 is the true accuracy of the worker
        //column 6 is the spammer score (RY) of the worker
        //column 7 is the worker cost (IPW) of the worker
        //*currently not used attributes
        public static void createIndividualSpamArffAUC(double[][] data, String name, double proportion) throws Exception
        {
            DecimalFormat df = new DecimalFormat("#.#####");
            File f = new File("SpammerArffs\\" + name + "\\" + name + ".arff");
            f.delete();
            BufferedWriter bw = new BufferedWriter(new FileWriter(f));
            bw.write("@RELATION " + name + "Spam\n\n");
            bw.write("@ATTRIBUTE proportion NUMERIC\n");
            bw.write("@ATTRIBUTE evenness NUMERIC\n");
            //bw.write("@ATTRIBUTE numclasses NUMERIC\n");
            bw.write("@ATTRIBUTE logsim NUMERIC\n");
            bw.write("@ATTRIBUTE emauc NUMERIC\n");
            bw.write("@ATTRIBUTE spamscore NUMERIC\n");
            bw.write("@ATTRIBUTE cost NUMERIC\n");
            bw.write("@ATTRIBUTE class {0,1}\n\n");
            //bw.write("@ATTRIBUTE class NUMERIC\n\n");
            bw.write("@DATA\n\n");
            double prop = 0;
            double threshold = 0;
            double[] nums = new double[data.length];
            for(int i = 0; i < data.length; i++)
            {
                nums[i] = data[i][4];
            }
            if(proportion == -1)
            {
                prop = .80;
                threshold = .80;
            }
            else
            {
                //threshold = StatCalc.findPercentile(nums, proportion);
                threshold = proportion;
            }
            for(int i = 0; i < data.length; i++)
            {
                boolean record = false;
                for(int j = 0; j < data[i].length; j++)
                {
                    if(data[i][j] == 0)
                        continue;
                    else
                    {
                        record = true;
                        break;
                    }
                }
                if(record)
                {
                    for(int j = 0; j < data[i].length; j++)
                    {
                        if(j != 2 && j != 4 && !(j == 5 || j == 6))
                            bw.write(df.format(data[i][j]) + ",");
                        if(j == 5 || j == 6)
                            //bw.write("" + 0 + ",");
                             bw.write(df.format(data[i][j]) + ",");
                        if(j == 4)
                            bw.write("" + 0 + ",");
                    }
                    
                    
                    if(data[i][4] > threshold)
                    //if(data[i][5] == 0)
                        bw.write("0\n");
                    else
                        bw.write("1\n");
                    
                    //For regression
                    //bw.write(df.format(data[i][5]) + "\n");
                }
            }
            bw.close();
        }
        
        public static Dataset integratedDataset(Dataset others, ArrayList<String> names, ArrayList<Dataset> datasets, String name, double threshold)
            throws Exception
        {
            ArrayList<Double> accs = new ArrayList();
            for(int i = 0; i < datasets.size(); i++)
            {
                if(name.equals(names.get(i)))
                    continue;
                Dataset dataset = datasets.get(i);
                ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
                ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
                WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
                for(int j = 0; j < workers.size(); j++)
                {
                    //if(((double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
			//			/ (double)graph.getTotalNumLabels() < .001) || (double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
                    //                    < 9)
                     //               continue;
                    accs.add(graph.getWorkerAccuracy(workers.get(j)));
                }
            }
            Collections.sort(accs);
            double threshVal = accs.get((int)((double)accs.size() * threshold));
            for(int i = 0; i < datasets.size(); i++)
            {
                if(name.equals(names.get(i)))
                    continue;
                Dataset dataset = datasets.get(i);
                Dataset copy = DatasetUtils.makeCopy(dataset);
                new DawidSkene(30).doInference(copy);
                ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(copy);
                ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(copy);
                WorkerTaskGraph graph = new WorkerTaskGraph(copy, workers, tasks);
                createIndividualSpamArff(graph.getWorkerData(),names.get(i),threshVal);
                Dataset d = FileLoader.loadFile("SpammerArffs\\" + names.get(i) + "\\" + names.get(i) + ".arff");
                //System.out.println("Adding " + names.get(i) + " to the mix, adding " + d.getExampleSize() + " examples.");
                Normalize n = new Normalize();
                n.setInputFormat(d);
                n.useFilter(d, n);
                others = DatasetUtils.combineDatasets(others, d);
            }
            //others.setClassIndex(others.numAttributes() - 1);
            others.setClassIndex(6);
            return others;
        }
        
        public static Dataset createSpammerDatasetFromAllOtherDatasets(ArrayList<Dataset> datasets, 
                ArrayList<String> names, String name, ArrayList<String> attributeNames, 
                String evalAttribute, double prop) throws Exception{
            Dataset combinedSpammerDataset = null;
            for(int i = 0; i < datasets.size(); i++){
                if(name.equals(names.get(i)))
                    continue;
                if(combinedSpammerDataset == null){
                    combinedSpammerDataset = FileLoader.loadFile(Analysis.createWorkerAttributesArffWithSpammerIndicatorLabel(datasets.get(i), names.get(i),
                        attributeNames, evalAttribute, prop));
                    Normalize n = new Normalize();
                    n.setInputFormat(combinedSpammerDataset);
                    n.useFilter(combinedSpammerDataset, n);
                }
                else{
                    Dataset anotherIndividualSpammerDataset = FileLoader.loadFile(Analysis.createWorkerAttributesArffWithSpammerIndicatorLabel(datasets.get(i), names.get(i),
                        attributeNames, evalAttribute, prop));
                    Normalize n = new Normalize();
                    n.setInputFormat(anotherIndividualSpammerDataset);
                    n.useFilter(anotherIndividualSpammerDataset, n);
                    combinedSpammerDataset = DatasetUtils.combineDatasets(combinedSpammerDataset, anotherIndividualSpammerDataset);
                }
            }
            return combinedSpammerDataset;
        }
        
        public static Dataset integratedDatasetAUC(Dataset others, ArrayList<String> names, ArrayList<Dataset> datasets, String name, double threshold)
            throws Exception
        {
            ArrayList<Double> AUCs = new ArrayList();
            for(int i = 0; i < datasets.size(); i++)
            {
                if(name.equals(names.get(i)))
                    continue;
                Dataset dataset = datasets.get(i);
                ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
                ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
                WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
                for(int j = 0; j < workers.size(); j++)
                {
                    //if(((double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
			//			/ (double)graph.getTotalNumLabels() < .001) || (double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
                    //                    < 9)
                     //               continue;
                    AUCs.add(graph.getWorkerEMAUC(workers.get(j)));
                }
            }
            Collections.sort(AUCs);
            double threshVal = AUCs.get((int)((double)AUCs.size() * threshold));
            for(int i = 0; i < datasets.size(); i++)
            {
                if(name.equals(names.get(i)))
                    continue;
                Dataset dataset = datasets.get(i);
                ArrayList<AnalyzedWorker> workers = SyntheticCrowdsourcing.getWorkersForDataset(dataset);
                ArrayList<AnalyzedTask> tasks = SyntheticCrowdsourcing.getTasksForDataset(dataset);
                WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
                WorkerTaskGraph wtg = new WorkerTaskGraph(datasets.get(i),SyntheticCrowdsourcing.getWorkersForDataset(datasets.get(i)),
                        SyntheticCrowdsourcing.getTasksForDataset(datasets.get(i)));
                createIndividualSpamArffAUC(wtg.getWorkerDataAUC(),names.get(i),threshVal);
                Dataset d = FileLoader.loadFile("SpammerArffs\\" + names.get(i) + "\\" + names.get(i) + ".arff");
                //System.out.println("Adding " + names.get(i) + " to the mix, adding " + d.getExampleSize() + " examples.");
                Normalize n = new Normalize();
                n.setInputFormat(d);
                n.useFilter(d, n);
                others = DatasetUtils.combineDatasets(others, d);
            }
            //others.setClassIndex(others.numAttributes() - 1);
            others.setClassIndex(6);
            return others;
        }
        
        public static Dataset simpleIntegratedDataset(Dataset others, ArrayList<String> names, ArrayList<Dataset> datasets, String name)
            throws Exception
        {
            for(int i = 0; i < datasets.size(); i++)
            {
                if(name.equals(names.get(i)))
                    continue;
                Dataset d = FileLoader.loadFile("SpammerArffs\\" + names.get(i) + "\\" + names.get(i) + ".arff");
                Normalize n = new Normalize();
                n.setInputFormat(d);
                n.useFilter(d, n);
                others = DatasetUtils.combineDatasets(others, d);
            }
            return others;
        }
        

}
