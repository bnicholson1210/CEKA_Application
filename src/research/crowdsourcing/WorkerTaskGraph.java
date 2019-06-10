package research.crowdsourcing;

import ceka.consensus.ds.DSWorker;
import ceka.consensus.ds.DawidSkene;
import java.util.ArrayList;
import java.util.Random;

import stat.StatCalc;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;
import ceka.utils.PerformanceStatistic;
import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import research.DatasetUtils;

public class WorkerTaskGraph 
{
	protected ArrayList<AnalyzedWorker> workers = new ArrayList();
	protected ArrayList<AnalyzedTask> tasks = new ArrayList();
	protected ArrayList<String[]> edges = new ArrayList();
	protected Dataset dataset;
        protected double[][][] workerCMs = null;
        protected double[][] workerSimMatrix = null;
	
	public WorkerTaskGraph(Dataset dataset, ArrayList<AnalyzedWorker> workers, 
			ArrayList<AnalyzedTask> tasks)
	{
		this.dataset = dataset;
		this.workers = workers;
		this.tasks = tasks;
		for(int i = 0; i < workers.size(); i++)
		{
			int numLabels = workers.get(i).getMultipleNoisyLabelSet(0).getLabelSetSize();
			for(int j = 0; j < numLabels; j++)
			this.addEdge(workers.get(i).getMultipleNoisyLabelSet(0).getLabel(j).getWorkerId(),
					workers.get(i).getMultipleNoisyLabelSet(0).getLabel(j).getExampleId());
		}
	}
        
        public WorkerTaskGraph(Dataset dataset)
        {
            this(dataset, SyntheticCrowdsourcing.getWorkersForDataset(dataset),
                    SyntheticCrowdsourcing.getTasksForDataset(dataset));
        }
        
        public ArrayList<AnalyzedWorker> getWorkers()
        {
            return workers;
        }
	
        public ArrayList<AnalyzedTask> getTasks(){
            return tasks;
        }
        
        public int getCategorySize(){
            return dataset.getCategorySize();
        }
        
	public void addEdge(String workerId, String taskId)
	{
		String[] edge = new String[2];
		edge[0] = workerId;
		edge[1] = taskId;
		edges.add(edge);
	}
	
	public ArrayList<AnalyzedTask> allTasksForWorker(AnalyzedWorker worker)
	{
		ArrayList<AnalyzedTask> tasks = new ArrayList();
		String workerId = worker.getId();
		for(String[] edge : edges)
		{
			if(edge[0].equals(workerId))
			{
				for(AnalyzedTask at : this.tasks)
				{
					if(at.getId().equals(edge[1]))
					{
						tasks.add(at);
					}
				}
			}
		}
		return tasks;
	}
	
	public ArrayList<AnalyzedWorker> allWorkersForTask(AnalyzedTask task)
	{
		ArrayList<AnalyzedWorker> workers = new ArrayList();
		String taskId = task.getId();
		for(String[] edge : edges)
		{
			if(edge[1].equals(taskId))
			{
				for(AnalyzedWorker w : this.workers)
				{
					if(w.getId().equals(edge[0]))
					{
						workers.add(w);
					}
				}
			}
		}
		return workers;
	}
	
        public ArrayList<AnalyzedTask> getAllTasksLabeledByBothWorkers(AnalyzedWorker w1, AnalyzedWorker w2){
            ArrayList<AnalyzedTask> tasks1 = allTasksForWorker(w1);
            ArrayList<AnalyzedTask> tasks2 = allTasksForWorker(w2);
            ArrayList<AnalyzedTask> resultTasks = new ArrayList();
            for(int i = 0; i < tasks1.size(); i++){
                for(int j = 0; j < tasks2.size(); j++){
                    if(tasks1.get(i).getId() == tasks2.get(j).getId())
                        resultTasks.add(tasks1.get(i));
                }
            }
            return resultTasks;
        }
        
        public double[] getBinarizedArrayOfWorkerLabelsForTasks(AnalyzedWorker worker, ArrayList<AnalyzedTask> tasks){
            double[] result = new double[tasks.size() * this.getCategorySize()];
            int index = 0;
            for(int i = 0; i < tasks.size(); i++){
                int label = this.labelFor(worker, tasks.get(i));
                for(int j = 0; j < this.getCategorySize(); j++){
                    if(j == label)
                        result[index] = 1;
                    else
                        result[index] = 0;
                    index++;
                }
            }
            return result;
        }
        
        public double[][] getBinarizedArrayOfTwoWorkersLabelsForTasks(AnalyzedWorker w1, AnalyzedWorker w2, ArrayList<AnalyzedTask> tasks){
            double[] res1 = getBinarizedArrayOfWorkerLabelsForTasks(w1, tasks);
            double[] res2 = getBinarizedArrayOfWorkerLabelsForTasks(w2, tasks);
            double[][] result = new double[2][res1.length];
            result[0] = res1;
            result[1] = res2;
            return result;
        }
        
        public double getProbabilityThatTwoWorkerLabelsWouldHappenByChance(AnalyzedWorker w1, AnalyzedWorker w2, ArrayList<AnalyzedTask> tasks){
            int sameCounter = 0;
            int numTasks = tasks.size();
            double probOfSame = 1.0 / (double)this.getCategorySize();
            for(int i = 0; i < tasks.size(); i++){
                AnalyzedTask task = tasks.get(i);
                int w1Label = this.labelFor(w1, task);
                int w2Label = this.labelFor(w2, task);
                if(w1Label == w2Label) sameCounter++;
            }
            //BigDecimal choose = StatCalc.choose(numTasks, sameCounter);
            //System.out.println("Choose: " + choose);
            BigDecimal prob = StatCalc.choose(numTasks, sameCounter).multiply(
                new BigDecimal(Math.pow(probOfSame, (double)sameCounter))).multiply(
                new BigDecimal(Math.pow(1.0 - probOfSame, (double)(numTasks - sameCounter))));
            return prob.doubleValue();
        }
        
        //first element: similarity, second element: confidence
        public double[] getSimilarityAndConfidenceOfTwoWorkers(AnalyzedWorker w1, AnalyzedWorker w2){
            double[] result = new double[2];
            result[0] = getSimilarityOfTwoWorkers(w1, w2);
            result[1] = getConfidenceOfTwoWorkers(w1, w2);
            return result;
        }
        
        public double getSimilarityOfTwoWorkers(AnalyzedWorker w1, AnalyzedWorker w2){
            ArrayList<AnalyzedTask> tasksBothHaveLabeled = this.getAllTasksLabeledByBothWorkers(w1, w2);
            double[][] binarizedArrays = getBinarizedArrayOfTwoWorkersLabelsForTasks(w1, w2, tasksBothHaveLabeled);
            double similarity = StatCalc.cosineSimilarity(binarizedArrays[0], binarizedArrays[1]);
            return similarity;
        }
        
        public double getConfidenceOfTwoWorkers(AnalyzedWorker w1, AnalyzedWorker w2){
            ArrayList<AnalyzedTask> tasksBothHaveLabeled = this.getAllTasksLabeledByBothWorkers(w1, w2);
            double confidence = 1.0 / this.getProbabilityThatTwoWorkerLabelsWouldHappenByChance(w1, w2, tasksBothHaveLabeled);
            return confidence;
        }
        
        public double getAverageSimilarityWithAllOtherWorkers(AnalyzedWorker w){
            double sum = 0;
            double numWorkersWhoSharedTasks = 0;
            for(int i = 0; i < workers.size(); i++){
                AnalyzedWorker other = workers.get(i);
                if(other.getId() == w.getId()) continue;
                ArrayList<AnalyzedTask> tasksBothHaveLabeled = this.getAllTasksLabeledByBothWorkers(w, other);
                if(tasksBothHaveLabeled.isEmpty()) continue;
                double similarity = getSimilarityOfTwoWorkers(w, other);
                sum += similarity;
                numWorkersWhoSharedTasks++;
            }
            if(numWorkersWhoSharedTasks == 0) return -1;
            else return sum / (double)numWorkersWhoSharedTasks;
        }
        
        public void printOutSomeStuffAboutAllCombinationsOfWorkersThatHasToDoWithHowSimilarAndGoodTheyAre(){
            ArrayList<AnalyzedWorker> workers = this.getWorkers();
            for(int i = 0; i < workers.size(); i++){
                AnalyzedWorker worker1 = workers.get(i);
                System.out.println("Worker " + i + " Accuracy: " + this.getWorkerAccuracy(worker1));
                System.out.println("Worker " + i + " AUC: " + this.getWorkerAUC(worker1));
                System.out.println("Worker " + i + " Average Similarity with Other Workers: " + this.getAverageSimilarityWithAllOtherWorkers(worker1));
                for(int j = 0; j < workers.size(); j++){
                    AnalyzedWorker worker2 = workers.get(j);
                    if(i != j){
                        ArrayList<AnalyzedTask> tasks = this.getAllTasksLabeledByBothWorkers(worker1, worker2);
                        if(tasks.isEmpty()) System.out.println("\tWorker " + i + " and Worker " + j + " shared no tasks.");
                        else{
                            double[] stuff = this.getSimilarityAndConfidenceOfTwoWorkers(worker1, worker2);
                            double similarity = stuff[0];
                            double confidence = stuff[1];
                            System.out.println("\tWorker " + i + " and Worker " + j + ":");
                            System.out.println("\t\tsimilarity: " + similarity);
                            System.out.println("\t\tconfidence: " + confidence);
                            System.out.println("\t\tproduct: " + similarity * confidence);
                        }
                    }
                }
            }
        }
        
        public void populateWorkerSimMatrix(){
            workerSimMatrix = new double[workers.size()][workers.size()];
            for(int i = 0; i < workers.size(); i++){
                for(int j = 0; j < workers.size(); j++){
                    if(i == j) workerSimMatrix[i][j] = 1;
                    else if(i > j) workerSimMatrix[i][j] = workerSimMatrix[j][i];
                    else if(this.getAllTasksLabeledByBothWorkers(workers.get(i), workers.get(j)).isEmpty())
                        workerSimMatrix[i][j] = -1;
                    else workerSimMatrix[i][j] = this.getSimilarityOfTwoWorkers(workers.get(i), workers.get(j));
                }
            }
        }
        
        public double[] getSimilarityValuesBetweenAllCombinationsOfWorkers(){
            ArrayList<Double> values = new ArrayList();
            if(this.workerSimMatrix == null) populateWorkerSimMatrix();
            for(int i = 0; i < workers.size(); i++){
                for(int j = i + 1; j < workers.size(); j++){
                    if(workerSimMatrix[i][j] == -1)
                        continue;
                    else values.add(workerSimMatrix[i][j]);
                }
            }
            double[] result = new double[values.size()];
            for(int i = 0; i < result.length; i++){
                result[i] = values.get(i);
            }
            return result;
        }
        
        //does the same as getSimilarityValuesBetweenAllCombinationsOfWorkers, but does not include
        //similarity values for pairs of workers where neither of them have a similarity
        //value with the worker w, or pairs of workers including w
        public double[] getSimilarityValuesBetweenAllCombinationsOfWorkersForWhichOneWorkerHasSimilarityValueWithWorker(AnalyzedWorker w){
            ArrayList<Double> values = new ArrayList();
            if(this.workerSimMatrix == null) populateWorkerSimMatrix();
            for(int i = 0; i < workers.size(); i++){
                if(this.getAllTasksLabeledByBothWorkers(w, workers.get(i)).isEmpty()) continue;
                else if(w.getId().equals(workers.get(i).getId())) continue;
                for(int j = i + 1; j < workers.size(); j++){
                    if(workerSimMatrix[i][j] == -1)
                        continue;
                    //condition not in original algorithm. who knows what will happen???
                    //else if(this.getAllTasksLabeledByBothWorkers(w, workers.get(j)).isEmpty()) continue;
                    else values.add(workerSimMatrix[i][j]);
                }
            }
            double[] result = new double[values.size()];
            for(int i = 0; i < result.length; i++){
                result[i] = values.get(i);
            }
            return result;
        }
        
        public double get3rdQuartileWorkerSimilarityValue(){
            return getNthQuartileWorkerSimilarityValue(3);
        }
        
        public double get1stQuartileWorkerSimilarityValue(){
            return getNthQuartileWorkerSimilarityValue(1);
        }
        
        public double getNthQuartileWorkerSimilarityValue(int n){
            return StatCalc.quartiles(this.getSimilarityValuesBetweenAllCombinationsOfWorkers())[n - 1];
        }
        
        public double getNthQuartileWorkerSimilarityValueOfRelatedWorkers(int n, AnalyzedWorker w){
            return StatCalc.quartiles(this.getSimilarityValuesBetweenAllCombinationsOfWorkersForWhichOneWorkerHasSimilarityValueWithWorker(w))[n - 1];
        }
        
        public void checkThatAllTasksHaveLabel(){
            for(int i = 0; i < tasks.size(); i++){
                if(this.allWorkersForTask(tasks.get(i)).isEmpty())
                    System.out.println("Task " + tasks.get(i).getId() + " does not have any labels!");
            }
        }
        
        public Dataset removeWorkersWithNoSimilarityPastNthQuartile(int n){
            ArrayList<AnalyzedWorker> workers = getWorkers();
            double nthQuartile = this.getNthQuartileWorkerSimilarityValue(n);
            ArrayList<AnalyzedWorker> remove = new ArrayList();
            for(int i = 0; i < workers.size(); i++){
                boolean pastNthQuartile = false;
                for(int j = i + 1; j < workers.size(); j++){
                    if(this.getAllTasksLabeledByBothWorkers(workers.get(i), workers.get(j)).isEmpty())
                        continue;
                    else{
                        if(this.getSimilarityOfTwoWorkers(workers.get(i), workers.get(j)) > nthQuartile){
                            pastNthQuartile = true;
                            break;
                        }
                    }
                }
                if(!pastNthQuartile){
                    remove.add(workers.get(i));
                }
            }
            System.out.println("removing " + remove.size() + " spammers.");
            return removeSpammers(remove);
        }
        
        public ArrayList<AnalyzedWorker> returnWorkersWithNoSimilarityPastNthQuartile(int n){
            ArrayList<AnalyzedWorker> workers = getWorkers();
            double nthQuartile = this.getNthQuartileWorkerSimilarityValue(n);
            ArrayList<AnalyzedWorker> remove = new ArrayList();
            for(int i = 0; i < workers.size(); i++){
                boolean pastNthQuartile = false;
                for(int j = i + 1; j < workers.size(); j++){
                    if(this.getAllTasksLabeledByBothWorkers(workers.get(i), workers.get(j)).isEmpty())
                        continue;
                    else{
                        if(this.getSimilarityOfTwoWorkers(workers.get(i), workers.get(j)) > nthQuartile){
                            pastNthQuartile = true;
                            break;
                        }
                    }
                }
                if(!pastNthQuartile){
                    remove.add(workers.get(i));
                }
            }
            //System.out.println("removing " + remove.size() + " spammers.");
            return remove;
        }
        
        public Dataset removeWorkersWithNoSimilarityPastNthQuartileOfRelatedWorkers(int n){
            ArrayList<AnalyzedWorker> workers = getWorkers();
            ArrayList<AnalyzedWorker> remove = new ArrayList();
            for(int i = 0; i < workers.size(); i++){
                if(i % 10 == 0 && i != 0) System.out.println();
                System.out.print(i + ",");
                double nthQuartile = this.getNthQuartileWorkerSimilarityValueOfRelatedWorkers(n, workers.get(i));
                boolean pastNthQuartile = false;
                for(int j = i + 1; j < workers.size(); j++){
                    if(this.getAllTasksLabeledByBothWorkers(workers.get(i), workers.get(j)).isEmpty())
                        continue;
                    else{
                        if(this.getSimilarityOfTwoWorkers(workers.get(i), workers.get(j)) > nthQuartile){
                            pastNthQuartile = true;
                            break;
                        }
                    }
                }
                if(!pastNthQuartile){
                    remove.add(workers.get(i));
                }
            }
            System.out.println("removing " + remove.size() + " spammers.");
            return removeSpammers(remove);
        }
        
        public Dataset removeWorkersWithNotAllSimilaritiesPastNthQuartileOfRelatedWorkers(int n){
            ArrayList<AnalyzedWorker> workers = getWorkers();
            ArrayList<AnalyzedWorker> remove = new ArrayList();
            for(int i = 0; i < workers.size(); i++){
                if(i % 10 == 0 && i != 0) System.out.println();
                System.out.print(i + ",");
                double nthQuartile = this.getNthQuartileWorkerSimilarityValueOfRelatedWorkers(n, workers.get(i));
                boolean allPastNthQuartile = true;
                for(int j = i + 1; j < workers.size(); j++){
                    if(this.getAllTasksLabeledByBothWorkers(workers.get(i), workers.get(j)).isEmpty())
                        continue;
                    else{
                        if(this.getSimilarityOfTwoWorkers(workers.get(i), workers.get(j)) <= nthQuartile){
                            allPastNthQuartile = false;
                            break;
                        }
                    }
                }
                if(!allPastNthQuartile){
                    remove.add(workers.get(i));
                }
            }
            System.out.println("removing " + remove.size() + " spammers.");
            return removeSpammers(remove);
        }
        
        public Dataset getDataset(){
            return dataset;
        }
        
	public int labelFor(AnalyzedWorker worker, AnalyzedTask task)
	{
                Label label = LabelFor(worker, task);
                if(label != null)
                    return label.getValue();
                else 
                    return -1;
	}
        
        public Label LabelFor(AnalyzedWorker worker, AnalyzedTask task)
	{
		int numLabels = worker.getMultipleNoisyLabelSet(0).getLabelSetSize();
		for(int i = 0; i < numLabels; i++)
		{
			Label l = worker.getMultipleNoisyLabelSet(0).getLabel(i);
			if(l.getExampleId().equals(task.getId()))
			{
				return l;
			}
		}
		return null;
	}
	
	public void evaluateAccuracyOfSimulation()
	{
		int totalLabels = 0;
		//Labels for which the random generation matches the 
		//label really given by worker to task
		int preciseLabels = 0;
		//Labels given by workers that match the true label of task
		int correctGivenLabels = 0;
		//Randomly generated labels that match the true label of task
		int correctRandomLabels = 0;
		//Discover the max number of labels given to any one task
		int max = Integer.MIN_VALUE;
		for(int i = 0; i < tasks.size(); i++)
		{
			int numLabels = allWorkersForTask(tasks.get(i)).size();
			if(numLabels > max)
				max = numLabels;
		}
		//Create an array to keep track of the "bag" accuracy for every task
		//where taskLabels[i][j][0] is the true given jth label of the ith task 
		//and   taskLabels[i][j][1] is the random jth label of the ith task
		int[][][] taskLabels = new int[tasks.size()][max][2];
		for(int i = 0; i < taskLabels.length; i++)
		{
			for(int j = 0; j < max; j++)
			{
				for(int k = 0; k < 2; k++)
				{
					taskLabels[i][j][k] = -1;
				}
			}
		}
		
		for(int i = 0; i < tasks.size(); i++)
		{
			AnalyzedTask thisTask = tasks.get(i);
			ArrayList<AnalyzedWorker> theseWorkers = allWorkersForTask(thisTask);
			for(int j = 0; j < theseWorkers.size(); j++)
			{
				AnalyzedWorker thisWorker = theseWorkers.get(j);
				int trueLabel = thisTask.getTrueLabel().getValue();
				totalLabels++;
				int generatedLabel = generateLabelFor(thisWorker, thisTask);
				int givenLabel = labelFor(thisWorker, thisTask);
				if(generatedLabel == givenLabel)
					preciseLabels++;
				if(givenLabel == trueLabel)
					correctGivenLabels++;
				if(generatedLabel == trueLabel)
					correctRandomLabels++;
				taskLabels[i][j][0] = givenLabel;
				taskLabels[i][j][1] = generatedLabel;
			}
		}
		System.out.println("Out of " + totalLabels + " total labels:\n\t" + (double)preciseLabels /
				(double) totalLabels + " of the random labels match the given labels\n\t"
				+ (double)correctRandomLabels / (double) totalLabels + " of the random labels"
				+ " match the true labels\n\t" + (double)correctGivenLabels / (double)totalLabels
				+ " of the given labels match the true labels");
		int bagCorrect = 0;
		for(int i = 0; i < tasks.size(); i++)
		{
			for(int j = 0; j < max; j++)
			{
				int value = taskLabels[i][j][0];
				if(value == -1)
				{
					break;
				}
				for(int k = 0; k < max; k++)
				{
					if(taskLabels[i][k][1] == value)
					{
						bagCorrect++;
						taskLabels[i][k][1] = Integer.MAX_VALUE;
						break;
					}
					if(taskLabels[i][k][1] == -1)
						break;
				}
			}
		}
		
		System.out.println("Bag accuracy: " + (double)bagCorrect / (double)totalLabels);
		
		//for(int i = 0; i < 15; i++)
		//{
		//	System.out.println("Instance " + i + " true label: " + dataset.getExampleByIndex(i).getTrueLabel().getValue());
		//}
	}
	
	public static int generateLabelFor(AnalyzedWorker worker, AnalyzedTask task)
	{
		int trueLabel = task.getTrueLabel().getValue();
		int genLabel = 0;
		double[] probs = new double[task.tendencies.length];
		for(int i = 0; i < probs.length; i++)
		{
			probs[i] = worker.tendencies[trueLabel][i] * task.tendencies[i];
			double denom = 0;
			for(int j = 0; j < probs.length; j++)
			{
				denom += worker.tendencies[trueLabel][j] * task.tendencies[j];
			}
			probs[i] /= denom;
		}
		double sum = 0;
		for(int i = 0; i < probs.length; i++)
		{
			sum += probs[i];
		}
		for(int i = 0; i < probs.length; i++)
		{
			probs[i] /= sum;
		}
		Random rand = new Random();
		double number = rand.nextDouble();
		for(int i = 0; i < probs.length; i++)
		{
			if(number <= probs[i])
			{
				genLabel = i;
				return genLabel;
			}
			else
				number -= probs[i];
		}
		return probs.length - 1;
	}
	
	public void inferWorkerAndTaskTendencies(int numCategories, int maxGuesses)
	{
		Random rand = new Random();
		//Only good for binary applications for now!!
		for(int i = 0; i < 5; i++)
		{
			updateTaskTendencies(numCategories, maxGuesses);
			updateWorkerTendencies(numCategories, maxGuesses);
		}
		
	}
	
	public void updateWorkerTendencies(int numCategories, int maxGuesses)
	{
		//Update each worker's tendencies
		//Worker tendencies have the form of a square matrix.
		//Assume there are two classes, so...
		
		//      (worker labels it...)
		//              0 1
		//(class is) 0 |0 1|   this means that for class 0, worker labels it 1 
		//(class is) 1 |1 0|   100% of the time, and vice versa.
		
		//int randNum = rand.nextInt() % workers.size();
		//int randNum = 52;
		
		Random rand = new Random();
		for(int i = 0; i < workers.size(); i++)
		{
			//System.out.println("" + i + "/" + (workers.size() - 1));
			AnalyzedWorker thisWorker = workers.get(i);
			ArrayList<AnalyzedTask> theseTasks = allTasksForWorker(thisWorker);
			//if(randNum == i)
				//System.out.println(thisWorker + " has labeled " + theseTasks.size() + " tasks.");
			//For each class:
			for(int c = 0; c < numCategories; c++)
			{
				//if(randNum == i)
					//System.out.println("For category " + c + ":");
				//Define maxProb for now to be the probability generated
				//by the current tendencies
				double maxProb = 1;
				{
					for(int j = 0; j < theseTasks.size(); j++)
					{
						AnalyzedTask thisTask = theseTasks.get(j);
						if(thisTask.getTrueLabel().getValue() == c)
						{
							/*
							if(randNum == i)
							{
								System.out.println("The worker has done " + thisTask
										+ "and has labeled it " + graph.labelFor(thisWorker, thisTask) + "\n(" + thisWorker.tendencies[c][graph.labelFor(thisWorker, thisTask)] +
										" x " + thisTask.tendencies[graph.labelFor(thisWorker, thisTask)] + ") / (" +
										thisWorker.tendencies[c][graph.labelFor(thisWorker, thisTask)] +
										" x " + thisTask.tendencies[graph.labelFor(thisWorker, thisTask)] 
										+ " + " + (1 - thisWorker.tendencies[c][graph.labelFor(thisWorker, thisTask)])
										+ " x " + thisTask.tendencies[(graph.labelFor(thisWorker, thisTask) + 1) % 2]
										+ ")");
							}
							*/
							maxProb = maxProb * (thisWorker.tendencies[c][labelFor(thisWorker, thisTask)]
									* thisTask.tendencies[labelFor(thisWorker, thisTask)]);
							maxProb = maxProb / (thisWorker.tendencies[c][labelFor(thisWorker, thisTask)]
									* thisTask.tendencies[labelFor(thisWorker, thisTask)]
									+ (1 - thisWorker.tendencies[c][labelFor(thisWorker, thisTask)]) * 
									thisTask.tendencies[(labelFor(thisWorker, thisTask) + 1) % 2]);
							//if(randNum == i)
								//System.out.println("MaxProb is now " + maxProb);
						}
					}
				}						
				
				double maxGuess = thisWorker.tendencies[c][0];
				
				//Generate 1,000 guesses as to the maximally likely
				//set of tendencies and test them
				for(int g = 0; g < maxGuesses; g++)
				{
					double prob = 1;
					double guess = rand.nextDouble();
					double[] guesses = new double[2];
					guesses[0] = guess;
					guesses[1] = 1 - guess;

					for(int j = 0; j < theseTasks.size(); j++)
					{
						AnalyzedTask thisTask = theseTasks.get(j);
						if(thisTask.getTrueLabel().getValue() == c)
						{
							int label = labelFor(thisWorker, thisTask);
							prob = prob * (guesses[label] * 
									thisTask.tendencies[label]);
							prob = prob / (guesses[label] * 
									thisTask.tendencies[label]
									+ guesses[(label + 1) % 2] * 
									thisTask.tendencies[(label + 1) % 2]);
						}
					}
					
					if(prob > maxProb)
					{
						maxProb = prob;
						maxGuess = guess;
					}
				}
				
				//Finally update worker's tendencies
				thisWorker.tendencies[c][0] = maxGuess;
				thisWorker.tendencies[c][1] = 1 - maxGuess;
				
				//if(i == randNum)
					//System.out.println("Now the worker looks like this: " + thisWorker);
			}
			//System.out.println(thisWorker);
		}
	}
	
	public void updateTaskTendencies(int numCategories, int maxGuesses)
	{
		//Update each task's tendencies.
		//Task tendencies have the form of a vector or array.
		//Suppose there are 2 classes. Task tendencies will 
		//have the form:
		//|0 1|
		//Meaning a 0% tendency to be labeled as class 0 and a 100% 
		//tendency to be labeled as class 1.
		
		Random rand = new Random();
		for(int i = 0; i < tasks.size(); i++)
		{
			//System.out.println("" + i + "/" + (tasks.size() - 1));
			AnalyzedTask thisTask = tasks.get(i);
			int category = thisTask.getTrueLabel().getValue();
			ArrayList<AnalyzedWorker> theseWorkers = allWorkersForTask(thisTask);
			double prob = 1;
			double[] guesses = new double[2];
			guesses[0] = thisTask.tendencies[0];
			guesses[1] = 1 - guesses[0];
			for(int j = 0; j < theseWorkers.size(); j++)
			{
				AnalyzedWorker thisWorker = theseWorkers.get(j);
				int label = labelFor(thisWorker, thisTask);
				prob = prob * (thisWorker.tendencies[category][label] * guesses[label]);
				prob = prob / (thisWorker.tendencies[category][label] * guesses[label]
						+ thisWorker.tendencies[category][(label + 1) % 2] *
						guesses[(label + 1) % 2]);
			}
			double maxGuess = guesses[0];
			double maxProb = prob;
			for(int g = 0; g < maxGuesses; g++)
			{
				double guess = rand.nextDouble();
				prob = 1;
				guesses = new double[2];
				guesses[0] = guess;
				guesses[1] = 1 - guess;
				for(int j = 0; j < theseWorkers.size(); j++)
				{
					AnalyzedWorker thisWorker = theseWorkers.get(j);
					int label = labelFor(thisWorker, thisTask);
					prob = prob * (thisWorker.tendencies[category][label] * guesses[label]);
					prob = prob / (thisWorker.tendencies[category][label] * guesses[label]
							+ thisWorker.tendencies[category][(label + 1) % 2] *
							guesses[(label + 1) % 2]);
				}
				
				if(prob > maxProb)
				{
					maxProb = prob;
					maxGuess = guess;
				}
			}
			thisTask.tendencies[0] = maxGuess;
			thisTask.tendencies[1] = 1 - maxGuess;
			//System.out.println(thisTask);
		}
	}
	
	public void initializeTendencies()
	{
		//Initialize each task's tendencies
		for(int i = 0; i < tasks.size(); i++)
		{
			ArrayList<AnalyzedWorker> theseWorkers = allWorkersForTask(tasks.get(i));
			for(int j = 0; j < theseWorkers.size(); j++)
			{
				int label = labelFor(theseWorkers.get(j), tasks.get(i));
				tasks.get(i).tendencies[label]++;
			}
			double sum = 0;
			for(int j = 0; j < tasks.get(i).tendencies.length; j++)
			{
				sum += tasks.get(i).tendencies[j];
			}
			for(int j = 0; j < tasks.get(i).tendencies.length; j++)
			{
				tasks.get(i).tendencies[j] = (double)tasks.get(i).tendencies[j] / (double)sum;
			}
			//System.out.println(tasks.get(i).tendencies[0] + " " + tasks.get(i).tendencies[1]);
		}
		
		//Initialize each worker's tendencies
		for(int i = 0; i < workers.size(); i++)
		{
			AnalyzedWorker thisWorker = workers.get(i);
			ArrayList<AnalyzedTask> theseTasks = allTasksForWorker(thisWorker);
			for(int j = 0; j < theseTasks.size(); j++)
			{
				AnalyzedTask thisTask = theseTasks.get(j);
				int trueLabel = thisTask.getTrueLabel().getValue();
				int guessedLabel = labelFor(thisWorker, thisTask);
				thisWorker.tendencies[trueLabel][guessedLabel]++;
			}
			for(int j = 0; j < thisWorker.tendencies.length; j++)
			{
				int sum = 0;
				for(int k = 0; k < thisWorker.tendencies[j].length; k++)
				{
					sum += thisWorker.tendencies[j][k];
				}
				for(int k = 0; k < thisWorker.tendencies[j].length; k++)
				{
					thisWorker.tendencies[j][k] = (double)thisWorker.tendencies[j][k] / (double)sum;
				}
			}
		}	
	}
	
	public double getWorkerAccuracy(AnalyzedWorker worker)
	{
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(worker);
            int counter = 0;
            int correct = 0;
            for(AnalyzedTask t : tasks)
            {
                if(labelFor(worker, t) == t.getTrueLabel().getValue())
                        correct++;
                counter++;
            }
            return (double)correct / (double)counter;
	}
        
        public double getWorkerEMAccuracy(AnalyzedWorker worker)
        {
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(worker);
            int counter = 0;
            int correct = 0;
            for(AnalyzedTask t : tasks)
            {
                if(labelFor(worker, t) == t.getIntegratedLabel().getValue())
                        correct++;
                counter++;
            }
            return (double)correct / (double)counter;
        }
        
        public double getWorkerEMAUC(AnalyzedWorker worker)
        {
            Dataset aDataset = DatasetUtils.makeCopy(dataset);
            for(int i = 0; i < aDataset.getExampleSize(); i++)
            {
                aDataset.getExampleByIndex(i).setTrueLabel(dataset.getExampleByIndex(i).getIntegratedLabel());
            }
            ArrayList<AnalyzedWorker> notThisWorker = new ArrayList();
            for(int i = 0; i < workers.size(); i++)
            {
                boolean a = workers.get(i).equals(worker);
                if(!a)
                {
                    notThisWorker.add(workers.get(i));
                }
            }
            WorkerTaskGraph g = new WorkerTaskGraph(aDataset, workers, tasks);
            aDataset = g.removeSpammers(notThisWorker);
            ArrayList<Integer> indices = new ArrayList();
            for(int i = 0; i < aDataset.getExampleSize(); i++)
            {
                Example e = aDataset.getExampleByIndex(i);
                if(e.getMultipleNoisyLabelSet(0).getLabelSetSize() == 0)
                    indices.add(i);
            }
            for(int i = aDataset.getExampleSize() - 1; i >= 0; i--)
            {
                if(indices.size() > 0 && i == indices.get(indices.size() - 1))
                {
                    aDataset.simpleRemoveExampleByIndex(i);
                    indices.remove(indices.size() - 1);
                }
            }

            for(int i  = 0; i < aDataset.getExampleSize(); i++)
            {
                Example e = aDataset.getExampleByIndex(i);
                String lab = "" + e.getMultipleNoisyLabelSet(0).getLabel(0).getValue();
                e.setIntegratedLabel(new Label("", lab, "", ""));
            }
            int previousValue = 0;
            boolean allAreTheSame = true;
            for(int i = 0; i < aDataset.getExampleSize(); i++)
            {
                if(i == 0)
                    previousValue = aDataset.getExampleByIndex(i).getTrueLabel().getValue();
                else
                {
                    if(previousValue != aDataset.getExampleByIndex(i).getTrueLabel().getValue())
                    {
                        allAreTheSame = false;
                        break;
                    }
                    previousValue = aDataset.getExampleByIndex(i).getTrueLabel().getValue();
                }
            }
            PerformanceStatistic ps = new PerformanceStatistic();
            ps.stat(aDataset);
            if(allAreTheSame)
                return ps.getAccuracy();
            else
                return ps.getAUC();
            
        }
	
        public double getWorkerAUC(AnalyzedWorker worker)
        {
            Dataset aDataset = DatasetUtils.makeCopy(dataset);
            for(int i = 0; i < aDataset.getExampleSize(); i++)
            {
                aDataset.getExampleByIndex(i).setTrueLabel(dataset.getExampleByIndex(i).getTrueLabel());
            }
            ArrayList<AnalyzedWorker> notThisWorker = new ArrayList();
            for(int i = 0; i < workers.size(); i++)
            {
                boolean a = workers.get(i).equals(worker);
                if(!a)
                {
                    notThisWorker.add(workers.get(i));
                }
            }
            WorkerTaskGraph g = new WorkerTaskGraph(aDataset, workers, tasks);
            aDataset = g.removeSpammers(notThisWorker);
            ArrayList<Integer> indices = new ArrayList();
            for(int i = 0; i < aDataset.getExampleSize(); i++)
            {
                Example e = aDataset.getExampleByIndex(i);
                if(e.getMultipleNoisyLabelSet(0).getLabelSetSize() == 0)
                    indices.add(i);
            }
            for(int i = aDataset.getExampleSize() - 1; i >= 0; i--)
            {
                if(indices.size() > 0 && i == indices.get(indices.size() - 1))
                {
                    aDataset.simpleRemoveExampleByIndex(i);
                    indices.remove(indices.size() - 1);
                }
            }

            for(int i  = 0; i < aDataset.getExampleSize(); i++)
            {
                Example e = aDataset.getExampleByIndex(i);
                String lab = "" + e.getMultipleNoisyLabelSet(0).getLabel(0).getValue();
                e.setIntegratedLabel(new Label("", lab, "", ""));
            }
            int previousValue = 0;
            boolean allAreTheSame = true;
            for(int i = 0; i < aDataset.getExampleSize(); i++)
            {
                if(i == 0)
                    previousValue = aDataset.getExampleByIndex(i).getTrueLabel().getValue();
                else
                {
                    if(previousValue != aDataset.getExampleByIndex(i).getTrueLabel().getValue())
                    {
                        allAreTheSame = false;
                        break;
                    }
                    previousValue = aDataset.getExampleByIndex(i).getTrueLabel().getValue();
                }
            }
            PerformanceStatistic ps = new PerformanceStatistic();
            ps.stat(aDataset);
            if(allAreTheSame)
                return ps.getAccuracy();
            else
                return ps.getAUC();
            
        }
                
	public double getWorkerLabelEvenness(AnalyzedWorker worker)
	{
		int numClasses = dataset.getCategorySize();
		ArrayList<AnalyzedTask> tasks = allTasksForWorker(worker);
		double[] counters = new double[numClasses];
		for(AnalyzedTask t : tasks)
		{
			counters[labelFor(worker, t)]++;
		}
		counters = StatCalc.normalize(counters);
		double product = 1.0 / StatCalc.choose(numClasses, 2).doubleValue();
		//System.out.println(StatCalc.choose(numClasses, 2));
		double sum = 0;
		int count = 0;
		for(int i = 0; i < counters.length; i++)
		{
			for(int j = i + 1; j < counters.length; j++)
			{
				double temp = sum;
				sum += (1.0 - Math.abs(counters[i] - counters[j])) * Math.min(counters[i], counters[j]);
				//System.out.println(++count + "\t" + (sum - temp));
			}
		}
		return product * sum * numClasses;
	}
	
	public static double getTrueLabelEvenness(ArrayList<AnalyzedTask> tasks, int numClasses)
	{
		double[] counters = new double[numClasses];
		for(AnalyzedTask t : tasks)
		{
			counters[t.getTrueLabel().getValue()]++;
		}
		counters = StatCalc.normalize(counters);
		double product = 1.0 / StatCalc.choose(numClasses, 2).doubleValue();
		//System.out.println(StatCalc.choose(numClasses, 2));
		double sum = 0;
		int count = 0;
		for(int i = 0; i < counters.length; i++)
		{
			for(int j = i + 1; j < counters.length; j++)
			{
				double temp = sum;
				sum += (1.0 - Math.abs(counters[i] - counters[j])) * Math.min(counters[i], counters[j]);
				//System.out.println(++count + "\t" + (sum - temp));
			}
		}
		return product * sum * numClasses;
	}
        
        public double getWorkerRelativeEvenness(AnalyzedWorker worker)
        {
            int numClasses = dataset.getCategorySize();
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(worker);
            double[] counters = new double[numClasses];
            double[] datasetCounters = new double[numClasses];
            for(AnalyzedTask t : tasks)
            {
                counters[labelFor(worker, t)]++;
            }
            
            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
                datasetCounters[dataset.getExampleByIndex(i).getTrueLabel().getValue()]++;
            }
            
            counters = StatCalc.normalize(counters);
            datasetCounters = StatCalc.normalize(datasetCounters);
            double[] result = new double[numClasses];
            for(int i = 0; i < result.length; i++)
            {
                result[i] = counters[i] - datasetCounters[i];
            }
            double sum = 0;
            for(int i = 0; i < result.length; i++)
            {
                sum += Math.abs(result[i]);
            }
            sum /= 2.0;
            return 1 - sum;
        }
	
	public double getWorkerSimilarity(AnalyzedWorker worker)
	{
		ArrayList<AnalyzedTask> tasks = allTasksForWorker(worker);
		double total = 0;
		double same = 0;
		for(int i = 0; i < tasks.size(); i++)
		{
			AnalyzedTask task = tasks.get(i);
			int givenLabel = labelFor(worker, task);
                        ArrayList<AnalyzedWorker> workers = allWorkersForTask(task);
			int numLabels = workers.size();
			total += numLabels;
			for(int j = 0; j < numLabels; j++)
			{
				if(labelFor(workers.get(j), task) == givenLabel)
					same++;
			}
		}
		return same / total;
	}
        
        public double getWorkerLogSimilarity(AnalyzedWorker worker)
        {
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(worker);
            double total = 0;
            double same = 0;
            double logSum = 0;
            for(int i = 0; i < tasks.size(); i++)
            {
                AnalyzedTask task = tasks.get(i);
                int givenLabel = labelFor(worker, task);
                ArrayList<AnalyzedWorker> workers = allWorkersForTask(task);
                for(int j = 0; j < workers.size(); j++)
                {
                    AnalyzedWorker w = workers.get(j);
                    if(labelFor(w, task) == givenLabel)
                    {
                        same++;
                    }
                    total++;
                }
                double sameProp = same / total;
                logSum += Math.log(sameProp);
            }
            return -1 * logSum / (double)tasks.size();
        }
        public double getWorkerDifference(AnalyzedWorker worker, AnalyzedTask task)
        {
            ArrayList<AnalyzedWorker> workers = allWorkersForTask(task);
            int same = 0;
            int total = 0;
            for(int i = 0; i < workers.size(); i++)
            {
                if(labelFor(workers.get(i),task) == labelFor(worker,task))
                    same++;
                total++;
            }
            double sim = (double)same / (double) total;
            return 1.0 - sim;
        }
        
        public Dataset removeSpammers(List<AnalyzedWorker> spammers)
        {
            Dataset result = dataset.generateEmpty();
            for(int i = 0; i < dataset.getCategorySize(); i++)
            {
                result.addCategory(dataset.getCategory(i));
            }
            for(int i = 0; i < dataset.getWorkerSize(); i++)
            {
                boolean add = true;
                for(int j = 0; j < spammers.size(); j++)
                {
                    if(dataset.getWorkerByIndex(i).equals(spammers.get(j).getWorker()))
                    {
                        add = false;
                        break;
                    }
                }
                if(add)
                    result.addWorker(dataset.getWorkerByIndex(i));
            }
            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
                Example newExample = (Example)dataset.getExampleByIndex(i).copy();
                newExample.resetMultiNoisyLabelSet();
                result.addExample(newExample);
            }
            for(int i = 0; i < result.getWorkerSize(); i++)
            {
                for(int j = 0; j < result.getWorkerByIndex(i).getMultipleNoisyLabelSet(0).getLabelSetSize(); j++)
                {
                    Label l = result.getWorkerByIndex(i).getMultipleNoisyLabelSet(0).getLabel(j);
                    result.getExampleById(l.getExampleId()).addNoisyLabel(l);
                }
            }
            return result;
        }
        
        public void correctLabels(AnalyzedWorker w)
        {
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(w);
            for(int i = 0; i < tasks.size(); i++)
            {
                AnalyzedTask task = tasks.get(i);
                ArrayList<AnalyzedWorker> otherWorkers = allWorkersForTask(task);
                double max = Double.NEGATIVE_INFINITY;
                AnalyzedWorker maxWorker = null;
                for(int j = 0; j < otherWorkers.size(); j++)
                {
                    double sim = otherWorkers.get(j).getSim();
                    if(sim > max)
                    {
                        max = sim;
                        maxWorker = otherWorkers.get(j);
                    }
                }
                LabelFor(w, task).setValue(labelFor(maxWorker, task));
            }
        }
        
        public void updateWorkerSims()
        {
            for(int i = 0; i < workers.size(); i++)
            {
                workers.get(i).setSim(getWorkerSimilarity(workers.get(i)));
            }
        }
        
        public int getTotalNumLabels()
        {
            int sum = 0;
            for(int i = 0; i < workers.size(); i++)
            {
                sum += workers.get(i).getMultipleNoisyLabelSet(0).getLabelSetSize();
            }
            return sum;
        }
        
        public double getLabelQuality()
        {
            double corr = 0;
            double total = 0;
            for(int i = 0; i < workers.size(); i++)
            {
                AnalyzedWorker w = workers.get(i);
                ArrayList<AnalyzedTask> tasks = allTasksForWorker(w);
                for(int j = 0; j < tasks.size(); j++)
                {
                    AnalyzedTask t = tasks.get(j);
                    if(labelFor(w,t) == t.getTrueLabel().getValue())
                    {
                        corr++;
                    }
                    total++;
                }
            }
            return corr / total;
        }
        
        public double averageEvenness()
        {
            double sum = 0;
            for(int i = 0; i < workers.size(); i++)
            {
                sum += this.getWorkerLabelEvenness(workers.get(i));
            }
            return sum / (double)workers.size();
        }
        
        public Dataset removeWorkerWithHighestDeltaEntropy(){
            ArrayList<AnalyzedWorker> workers = getWorkers();
            AnalyzedWorker maxWorker = null;
            double max = Double.NEGATIVE_INFINITY;
            for(int i = 0; i < workers.size(); i++){
                AnalyzedWorker worker = workers.get(i);
                double deltaEnt = getWorkerDeltaEntropy(worker);
                if(deltaEnt > max){
                    max = deltaEnt;
                    maxWorker = worker;
                }
            }
            //System.out.println("Removing worker with entropy " + max);
            ArrayList<AnalyzedWorker> remove = new ArrayList();
            remove.add(maxWorker);
            return removeSpammers(remove);
        }
        
        public Dataset removeWorkerWithLowestDeltaEntropy(){
            ArrayList<AnalyzedWorker> workers = getWorkers();
            AnalyzedWorker minWorker = null;
            double min = Double.POSITIVE_INFINITY;
            for(int i = 0; i < workers.size(); i++){
                AnalyzedWorker worker = workers.get(i);
                double deltaEnt = getWorkerDeltaEntropy(worker);
                if(deltaEnt < min){
                    min = deltaEnt;
                    minWorker = worker;
                }
            }
            ArrayList<AnalyzedWorker> remove = new ArrayList();
            remove.add(minWorker);
            Dataset result = removeSpammers(remove);
            return result;
        }
        
        public Dataset removeWorkerWithHighestRelativeDeltaEntropy(){
             ArrayList<AnalyzedWorker> workers = getWorkers();
            AnalyzedWorker maxWorker = null;
            double max = Double.NEGATIVE_INFINITY;
            for(int i = 0; i < workers.size(); i++){
                AnalyzedWorker worker = workers.get(i);
                double deltaEnt = getWorkerRelativeDeltaEntropy(worker);
                if(deltaEnt > max){
                    max = deltaEnt;
                    maxWorker = worker;
                }
            }
            //System.out.println("Removing worker with entropy " + max);
            ArrayList<AnalyzedWorker> remove = new ArrayList();
            remove.add(maxWorker);
            return removeSpammers(remove);
        }
        
        public Dataset removeWorkerWithLowestRelativeDeltaEntropy(){
            ArrayList<AnalyzedWorker> workers = getWorkers();
            AnalyzedWorker minWorker = null;
            double min = Double.POSITIVE_INFINITY;
            for(int i = 0; i < workers.size(); i++){
                AnalyzedWorker worker = workers.get(i);
                double deltaEnt = getWorkerRelativeDeltaEntropy(worker);
                if(deltaEnt < min){
                    min = deltaEnt;
                    minWorker = worker;
                }
            }
            ArrayList<AnalyzedWorker> remove = new ArrayList();
            remove.add(minWorker);
            Dataset result = removeSpammers(remove);
            return result;
        }
        
        public Dataset removeWorkerWithLowestAverageSimilarityWithOtherWorkers(){
            ArrayList<AnalyzedWorker> workers = getWorkers();
            AnalyzedWorker minWorker = null;
            double min = Double.POSITIVE_INFINITY;
            for(int i = 0; i < workers.size(); i++){
                AnalyzedWorker worker = workers.get(i);
                //Absolute value because if worker did not share tasks with anyone then value is -1 but we don't
                //want to filter them
                double averageSim = Math.abs(this.getAverageSimilarityWithAllOtherWorkers(worker));
                if(averageSim < min){
                    min = averageSim;
                    minWorker = worker;
                }
            }
            System.out.println("Removing worker (" + minWorker.getId() + ") with accuracy: " + this.getWorkerAccuracy(minWorker) +
                    ", AUC: " + this.getWorkerAUC(minWorker) + ", average similarity: " + min);
            ArrayList<AnalyzedWorker> remove = new ArrayList();
            remove.add(minWorker);
            Dataset result = removeSpammers(remove);
            return result;
        }
        
        public double getWorkerDeltaEntropy(AnalyzedWorker w){
            ArrayList<AnalyzedTask> workerTasks = allTasksForWorker(w);
            //iterate through the tasks to get the change in entropy for each task
            double averageWorkerDeltaEntropy = 0;
            for(int i = 0; i < workerTasks.size(); i++)
            {
                AnalyzedTask task = workerTasks.get(i);
                ArrayList<AnalyzedWorker> workers = allWorkersForTask(task);
                //iterate through all the workers that have completed this task
                //to get the entropy of the task with and without the worker
                double entropyWithoutWorker = 0;
                double entropyWithWorker = 0;
                HashMap<String, Integer> labelCountsWith = new HashMap();
                HashMap<String, Integer> labelCountsWithout = new HashMap();
                for(int j = 0; j < workers.size(); j++)
                {
                    AnalyzedWorker worker = workers.get(j);
                    int label = labelFor(worker, task);
                    Integer count = labelCountsWith.get("" + label);
                    if(count == null) labelCountsWith.put("" + label, new Integer(1));
                    else labelCountsWith.put("" + label, new Integer(count + 1));
                    if(!w.equals(worker)){
                        Integer c = labelCountsWithout.get("" + label);
                        if(c == null) labelCountsWithout.put("" + label, new Integer(1));
                        else labelCountsWithout.put("" + label, new Integer(c + 1));
                    }
                }
                Iterator<String> withIterator = labelCountsWith.keySet().iterator();
                double withEntropy = 0;
                while(withIterator.hasNext()){
                    String key = withIterator.next();
                    int numWorkers = workers.size();
                    double p = (double)labelCountsWith.get(key).intValue() / (double)numWorkers;
                    withEntropy += -1 * p * Math.log(p);
                }
                Iterator<String> withoutIterator = labelCountsWithout.keySet().iterator();
                double withoutEntropy = 0;
                while(withoutIterator.hasNext()){
                    String key = withoutIterator.next();
                    int numWorkers = workers.size() - 1;
                    double p = (double)labelCountsWithout.get(key).intValue() / (double)numWorkers;
                    withoutEntropy += -1 * p * Math.log(p);
                }
                double deltaEntropy = withEntropy - withoutEntropy;
                averageWorkerDeltaEntropy += deltaEntropy;
            }
            averageWorkerDeltaEntropy /= (double)workerTasks.size();
            return averageWorkerDeltaEntropy;
        }
        
        public double getWorkerRelativeDeltaEntropy(AnalyzedWorker w){
            ArrayList<AnalyzedTask> workerTasks = allTasksForWorker(w);
            //iterate through the tasks to get the change in entropy for each task
            double averageWorkerDeltaEntropy = 0;
            HashMap<String, Integer> totalLabelCounts = new HashMap();
            for(int i = 0; i < workers.size(); i++){
                for(int j = 0; j < tasks.size(); j++){
                    AnalyzedWorker worker = workers.get(i);
                    AnalyzedTask task = tasks.get(j);
                    int label = labelFor(worker, task);
                    if(label != -1){
                        Integer count = totalLabelCounts.get("" + label);
                        if(count == null) totalLabelCounts.put("" + label, new Integer(1));
                        else totalLabelCounts.put("" + label, new Integer(count + 1));
                    }
                }
            }
            for(int i = 0; i < workerTasks.size(); i++)
            {
                AnalyzedTask task = workerTasks.get(i);
                ArrayList<AnalyzedWorker> workers = allWorkersForTask(task);
                //iterate through all the workers that have completed this task
                //to get the entropy of the task with and without the worker
                double entropyWithoutWorker = 0;
                double entropyWithWorker = 0;
                HashMap<String, Integer> labelCountsWith = new HashMap();
                HashMap<String, Integer> labelCountsWithout = new HashMap();
                for(int j = 0; j < workers.size(); j++)
                {
                    AnalyzedWorker worker = workers.get(j);
                    int label = labelFor(worker, task);
                    Integer count = labelCountsWith.get("" + label);
                    if(count == null) labelCountsWith.put("" + label, new Integer(1));
                    else labelCountsWith.put("" + label, new Integer(count + 1));
                    if(!w.equals(worker)){
                        Integer c = labelCountsWithout.get("" + label);
                        if(c == null) labelCountsWithout.put("" + label, new Integer(1));
                        else labelCountsWithout.put("" + label, new Integer(c + 1));
                    }
                }
                HashMap<String, Double> withDist = StatCalc.getRelativeDistribution(totalLabelCounts, labelCountsWith);
                //System.out.println("with dist: " + withDist);
                HashMap<String, Double> withoutDist = StatCalc.getRelativeDistribution(totalLabelCounts, labelCountsWithout);
                //System.out.println("without dist: " + withoutDist);
                double withEntropy = StatCalc.entropy(withDist);
                //System.out.println("with entropy: " + withEntropy);
                double withoutEntropy = StatCalc.entropy(withoutDist);
                //System.out.println("without entropy: " + withoutEntropy);
                double deltaEntropy = withEntropy - withoutEntropy;
                //System.out.println("Resulting delta entropy: " + deltaEntropy);
                averageWorkerDeltaEntropy += deltaEntropy;
            }
            averageWorkerDeltaEntropy /= (double)workerTasks.size();
            return averageWorkerDeltaEntropy;
        }
        
        public double getTaskEntropy(AnalyzedTask task){
            ArrayList<AnalyzedWorker> workers = allWorkersForTask(task);
            //iterate through all the workers that have completed this task
            //to calculate the entropy of the labels for this task
            double entropyWithWorker = 0;
            HashMap<String, Integer> labelCountsWith = new HashMap();
            for(int j = 0; j < workers.size(); j++)
            {
                AnalyzedWorker worker = workers.get(j);
                int label = labelFor(worker, task);
                Integer count = labelCountsWith.get("" + label);
                if(count == null) labelCountsWith.put("" + label, new Integer(1));
                else labelCountsWith.put("" + label, new Integer(count + 1));
            }
            Iterator<String> withIterator = labelCountsWith.keySet().iterator();
            double withEntropy = 0;
            while(withIterator.hasNext()){
                String key = withIterator.next();
                int numWorkers = workers.size();
                double p = (double)labelCountsWith.get(key).intValue() / (double)numWorkers;
                withEntropy += -1 * p * Math.log(p);
            }
            return withEntropy;
        }
        
        //returns the average task entropy of the dataset 
        public double getDatasetEntropy(){
            ArrayList<AnalyzedTask> allTasks = getTasks();
            double sum = 0;
            for(int i = 0; i < allTasks.size(); i++){
                sum += getTaskEntropy(allTasks.get(i));
            }
            return sum / (double)allTasks.size();
        }
        
        public double[] maxAndMinLogSims()
        {
            double max = Double.NEGATIVE_INFINITY;
            double min = Double.POSITIVE_INFINITY;
            for(int i = 0; i < workers.size(); i++)
            {
                AnalyzedWorker w = workers.get(i);
                double sim = this.getWorkerLogSimilarity(w);
                if(sim > max)
                    max = sim;
                if(sim < min)
                    min = sim;
            }
            double[] result = {max, min};
            return result;
        }
        
        public double[][][] getWorkerCMs()
        {
            if(workerCMs == null)
            {
                DawidSkene ds = new DawidSkene(30);
                ds.doInference(dataset);
                int numCategories = dataset.getCategorySize();
                ArrayList<DSWorker> workers = ds.getWorkers();
                int numWorkers = workers.size();
                double[][][] result = new double[numWorkers][numCategories][numCategories];
                for(int i = 0; i < workers.size(); i++)
                {
                    result[i] = workers.get(i).getCM();
                }
                workerCMs = result;
                return result;
            }
            else
                return workerCMs;
        }
        
        //Creates a dataset identical to the dataset of this graph,
        //except that all workers chose the majority class as all their labels.
        //This is a function because of calculating the threshold for some
        //spam-elimination filters.
        public Dataset generateMajoritySpammerDataset()
        {
            Dataset newDataset = dataset.generateEmpty();
            double[] labelCounts = new double[dataset.getCategorySize()];
            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
                labelCounts[dataset.getExampleByIndex(i).getTrueLabel().getValue()]++;
            }
            int majorityLabel = StatCalc.maxIndex(labelCounts);
            for(int i = 0; i < dataset.getCategorySize(); i++)
            {
                newDataset.addCategory(dataset.getCategory(i));
            }
            

            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
                newDataset.addExample((Example)dataset.getExampleByIndex(i).copy());
                newDataset.getExampleByIndex(i).setTrueLabel(dataset.getExampleByIndex(i).getTrueLabel());
                int size = dataset.getExampleByIndex(i).getMultipleNoisyLabelSet(0).getLabelSetSize();
                for(int j = 0; j < size; j++)
                {
                    Label copy = dataset.getExampleByIndex(i).getMultipleNoisyLabelSet(0).getLabel(j).copy();
                    copy.setValue(majorityLabel);
                    newDataset.getExampleByIndex(i).addNoisyLabel(copy);
                }
            }
            for(int i = 0; i < workers.size(); i++)
            {
                Worker w = new Worker(workers.get(i).getId());
                for(int j = 0; j < workers.get(i).getMultipleNoisyLabelSet(0).getLabelSetSize(); j++)
                {
                    Label copy = workers.get(i).getMultipleNoisyLabelSet(0).getLabel(j).copy();
                    copy.setValue(majorityLabel);
                    w.addNoisyLabel(copy);
                }
                newDataset.addWorker(w);
            }
            return newDataset;
        }
        
        //Returns true if w has labeled any tasks of true class c, false otherwise
        public boolean anyLabelsForTrueClass(AnalyzedWorker w, int c)
        {
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(w);
            for(int i = 0; i < tasks.size(); i++)
            {
                if(tasks.get(i).getTrueLabel().getValue() == c)
                    return true;
            }
            return false;
        }
        
        //Returns the prior probability that the worker w labels a task as c
        public double workerPrior(AnalyzedWorker w, int c)
        {
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(w);
            int num = 0;
            int total = 0;
            for(int i = 0; i < tasks.size(); i++)
            {
                if(tasks.get(i).getTrueLabel().getValue() == c)
                    num++;
                total++;
            }
            return (double)num / (double)total;
        }
        
        public int getNumTasksForWorker(AnalyzedWorker w)
        {
            return allTasksForWorker(w).size();
        }
        
        public double[][] getWorkerData()
        {
            double[][] workerData2 = new double[workers.size()][7];
            double totalLabels = this.getTotalNumLabels();
            double averageEvenness = this.averageEvenness();
            for(int j = 0; j < workers.size(); j++)
            {
                //if(((double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
                //                                / (double)totalLabels < .001) || (double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
                 //                       < 9)
                 //                   continue;
                workerData2[j][0] = (double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize() /
                                (double)totalLabels;
                workerData2[j][1] = Math.abs(averageEvenness - this.getWorkerLabelEvenness(workers.get(j)));
                //workerData2[localWorkerNum][2] = dataset.getCategorySize();
                workerData2[j][3] = this.getWorkerLogSimilarity(workers.get(j));
                workerData2[j][4] = this.getWorkerEMAccuracy(workers.get(j));
                //workerData2[j][5] = this.getWorkerAccuracy(workers.get(j));
                workerData2[j][5] = this.spammerScore(workers.get(j), null)[0];
                workerData2[j][6] = this.workerCost(workers.get(j), null)[0];
            }
            return workerData2;
        }
        
        public double[][] getWorkerDataAUC()
        {
            double[][] workerData2 = new double[workers.size()][7];
            double totalLabels = this.getTotalNumLabels();
            double averageEvenness = this.averageEvenness();
            for(int j = 0; j < workers.size(); j++)
            {
                //if(((double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
                //                                / (double)totalLabels < .001) || (double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
                 //                       < 9)
                 //                   continue;
                workerData2[j][0] = (double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize() /
                                (double)totalLabels;
                workerData2[j][1] = Math.abs(averageEvenness - this.getWorkerLabelEvenness(workers.get(j)));
                //workerData2[localWorkerNum][2] = dataset.getCategorySize();
                workerData2[j][3] = this.getWorkerLogSimilarity(workers.get(j));
                workerData2[j][4] = this.getWorkerEMAUC(workers.get(j));
                //workerData2[j][5] = this.getWorkerAccuracy(workers.get(j));
                workerData2[j][5] = this.spammerScore(workers.get(j), null)[0];
                workerData2[j][6] = this.workerCost(workers.get(j), null)[0];
            }
            return workerData2;
        }
        
        //The first element is the spammer score for the worker, the second element
        //is the expected value for spam score if the worker were to label all tasks as belonging to majority class
        public double[] spammerScore(AnalyzedWorker w, double[][][] confusionMatrices)
        {
            if(confusionMatrices == null)
            {
                confusionMatrices = this.getWorkerCMs();
            }
            double score = 0;
            double spamScore = 0;
            int i = this.getWorkerIndex(w);
            int numClasses = dataset.getCategorySize();
            double[] labelCounts = new double[dataset.getCategorySize()];
            for(int j = 0; j < dataset.getExampleSize(); j++)
            {
                labelCounts[dataset.getExampleByIndex(j).getTrueLabel().getValue()]++;
            }
            int majorityLabel = StatCalc.maxIndex(labelCounts);
            for(int j = 0; j < numClasses; j++)
            {
                for(int k = 0; k < j; k++)
                {
                    for(int l = 0; l < numClasses; l++)
                    {
                        double scoreAdd = 1.0 / (numClasses * (numClasses - 1)) * Math.pow(confusionMatrices[i][k][l] - confusionMatrices[i][j][l], 2.0);
                        //double spamScoreAdd = 1.0 / (numClasses * (numClasses - 1)) * Math.pow(confusionMatrices2[i][k][l] - confusionMatrices2[i][j][l], 2.0);
                        if(scoreAdd < Double.POSITIVE_INFINITY && scoreAdd > Double.NEGATIVE_INFINITY)
                            score += scoreAdd;
                        //if(spamScoreAdd < Double.POSITIVE_INFINITY && scoreAdd > Double.NEGATIVE_INFINITY)
                            //spamScore += spamScoreAdd;
                        if(l == majorityLabel)
                        {
                            boolean a = this.anyLabelsForTrueClass(w,k);
                            boolean b = this.anyLabelsForTrueClass(w,j);
                            if((a && !b) || (!a && b))
                                spamScore += 1.0 / (numClasses * (numClasses - 1));
                        }
                    }
                }
            }
            double[] result = new double[2];
            result[0] = score;
            result[1] = spamScore;
            return result;
        }
        
        public HashMap getMostCommonLabels(){
            int numOfTasks = getTasks().size();
            HashMap<String, Integer> result = new HashMap();
//            int result[] = new int[numOfTasks];           
            for(int n = 0; n < numOfTasks; n++){ 
                AnalyzedTask task = getTasks().get(n);                
                ArrayList<AnalyzedWorker> associatedWorkers = allWorkersForTask(task);
                HashMap<String, Integer> labelCounts = new HashMap();
                for(int k = 0; k < associatedWorkers.size(); k++){
                    int label = labelFor(associatedWorkers.get(k), task);
                    Integer count = labelCounts.get(""+label);
                    if(count == null) labelCounts.put(""+label,new Integer(1));
                    else labelCounts.put(""+label,new Integer(count+1));
                }
                int biggestVal = 0;
                String mostCommon = "";
                for(HashMap.Entry<String,Integer> entry : labelCounts.entrySet()){
                    if(entry.getValue() > biggestVal){                        
                        biggestVal = entry.getValue();
                        mostCommon = entry.getKey();
                    }
                }
                result.put(""+task.getId(),Integer.parseInt(mostCommon));
            }
            return result;
        }
	public HashMap getLeastCommonLabels(){
            int numOfTasks = getTasks().size();
            HashMap<String, Integer> result = new HashMap();
//            int result[] = new int[numOfTasks];           
            for(int n = 0; n < numOfTasks; n++){ 
                AnalyzedTask task = getTasks().get(n);                
                ArrayList<AnalyzedWorker> associatedWorkers = allWorkersForTask(task);
                HashMap<String, Integer> labelCounts = new HashMap();
                for(int k = 0; k < associatedWorkers.size(); k++){
                    int label = labelFor(associatedWorkers.get(k), task);
                    Integer count = labelCounts.get(""+label);
                    if(count == null) labelCounts.put(""+label,new Integer(1));
                    else labelCounts.put(""+label,new Integer(count+1));
                }
                int lowestVal = 1000000;
                String leastCommon = "";
                for(HashMap.Entry<String,Integer> entry : labelCounts.entrySet()){
                    if(entry.getValue() < lowestVal){                        
                        lowestVal = entry.getValue();
                        leastCommon = entry.getKey();
                    }
                }
                result.put(""+task.getId(),Integer.parseInt(leastCommon));
            }
            return result;
        }
	
        public HashMap getKeyMap(){
            int numOfTasks = getTasks().size();
            HashMap<String, Integer> result = new HashMap();
            for (int n = 0; n < numOfTasks; n++){
                AnalyzedTask task = getTasks().get(n);
                result.put(""+task.getId(), task.getTrueLabel().getValue());
            }
            return result;
        }
        public int getWorkerIndex(AnalyzedWorker w)
        {
            for(int i = 0; i < workers.size(); i++)
            {
                if(w.equals(workers.get(i)))
                    return i;
            }
            return -1;
        }
        
        public double[] workerCost(AnalyzedWorker w, double[][][] confusionMatrices)
        {
            if(confusionMatrices == null)
            {
                confusionMatrices = this.getWorkerCMs();
            }
            ArrayList<ArrayList<ArrayList<Double>>> softLabels = new ArrayList();
            double[] labelCounts = new double[dataset.getCategorySize()];
            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
                labelCounts[dataset.getExampleByIndex(i).getTrueLabel().getValue()]++;
            }
            labelCounts = StatCalc.normalize(labelCounts);
            for(int i = 0; i < workers.size(); i++)
            {
                softLabels.add(new ArrayList());
                ArrayList<AnalyzedTask> t = this.allTasksForWorker(workers.get(i));
                for(int j = 0; j < t.size(); j++)
                {
                    softLabels.get(i).add(new ArrayList());
                }
            }
            ArrayList<AnalyzedTask> ts = this.allTasksForWorker(w);
            double workerCost = 0;
            double spammerCost = 0;
            int numClasses = dataset.getCategorySize();
            for(int j = 0; j < ts.size(); j++)
            {
                AnalyzedTask t = ts.get(j);
                //generate soft label (real and spammer)
                int label = this.labelFor(w, t);
                double[] softLabel = new double[numClasses];
                double[] spamSoftLabel = new double[numClasses];
                for(int k = 0; k < softLabel.length; k++)
                {
                    softLabel[k] = confusionMatrices[getWorkerIndex(w)][k][label] * labelCounts[k];
                    spamSoftLabel[k] = labelCounts[k];
                    softLabels.get(getWorkerIndex(w)).get(j).add(softLabel[k]);
                }
                softLabel = StatCalc.normalize(softLabel);
                spamSoftLabel = StatCalc.normalize(spamSoftLabel);
                //calculate cost of each label
                double cost = 0;
                double spamCost = 0;
                for(int k = 0; k < numClasses; k++)
                {
                    for(int l = 0; l < numClasses; l++)
                    {
                        if(k != l)
                        {
                            double val1 = softLabel[k] * softLabel[l];
                            if(val1 > Double.NEGATIVE_INFINITY && val1 < Double.POSITIVE_INFINITY)
                                cost += val1;
                            double val2 = spamSoftLabel[k] * spamSoftLabel[l];
                            if(val2 > Double.NEGATIVE_INFINITY && val2 < Double.POSITIVE_INFINITY)
                                spamCost += val2;
                        }
                    }
                }
                double val1 = cost * this.workerPrior(w, label);
                if(val1 > Double.NEGATIVE_INFINITY && val1 < Double.POSITIVE_INFINITY)
                      workerCost += val1;  
                double val2 = spamCost;
                if(val2 > Double.NEGATIVE_INFINITY && val2 < Double.POSITIVE_INFINITY)
                    spammerCost += val2;
            }
                double[] result = new double[2];
                result[0] = workerCost;
                result[1] = spammerCost;
                return result;
        }
        
        public Double getCharacteristicValueForWorker(String characteristic, AnalyzedWorker w)
        	throws Exception
        {
        	if(characteristic.toLowerCase().equals("distanceFromAverageEvenness".toLowerCase()))
        		return Math.abs(this.getWorkerLabelEvenness(w) - this.averageEvenness());
        	else if(characteristic.toLowerCase().equals("logSimilarity".toLowerCase()))
        		return this.getWorkerLogSimilarity(w);
        	else if(characteristic.toLowerCase().equals("EMAccuracy".toLowerCase()))
        		return this.getWorkerEMAccuracy(w);
        	else if(characteristic.toLowerCase().equals("Accuracy".toLowerCase()))
        		return this.getWorkerAccuracy(w);
                else if(characteristic.toLowerCase().equals("AUC".toLowerCase()))
                        return this.getWorkerAUC(w);
                    else if(characteristic.toLowerCase().equals("EMAUC".toLowerCase()))
                        return this.getWorkerEMAUC(w);
        	else if(characteristic.toLowerCase().equals("spammerScore".toLowerCase()))
        		return this.spammerScore(w, null)[0];
        	else if(characteristic.toLowerCase().equals("workerCost".toLowerCase()))
        		return this.workerCost(w,  null)[0];
        	else if(characteristic.toLowerCase().equals("proportion".toLowerCase()))
        		return (double)(w.getMultipleNoisyLabelSet(0).getLabelSetSize()) /
                        (double)this.getTotalNumLabels();
        	else
        		throw new Exception("The characteristic \"" + characteristic +  
        				"\" is not recognized.");
        }
        
        public String getHTMLOfDataset(){
            String result = "<!DOCTYPE html>\n" +
                "<html>\n" +
                "<head>\n" +
                "</head>\n" +
                "<body>\n" +
                "<table border=\"1\">\n";
            String rows = "";
            String headerRow = "<tr>\n";
            String footerRow = "<tr>\n";
            for(int i = 0; i < this.workers.size(); i++){
                AnalyzedWorker worker = this.workers.get(i);
                String row = "<tr>\n";
                for(int j = 0; j < this.tasks.size(); j++){
                    AnalyzedTask task = this.tasks.get(j);
                    if(i == 0){
                        if(j == 0){
                            headerRow += "<td></td>\n";
                            footerRow += "<td>True Label</td>\n";
                        }
                        headerRow += "<td>Task " + (j + 1) + "</td>\n";
                        footerRow += "<td>" + task.getTrueLabel().getValue() + "</td>\n";
                    }
                    if(j == 0)
                        row += "<td>Worker " + (i + 1) + "</td>\n";
                    int label = this.labelFor(worker, task);
                    row += "<td>" +  (label != -1 ? label : "") + "</td>\n";
                }
                //get spammer score of this worker [0], and spammer score of spammer who labeled worker's tasks [1]
                double[] spamScore = this.spammerScore(worker, null);
                double entropy = this.getWorkerRelativeDeltaEntropy(worker);
                double acc = this.getWorkerAccuracy(worker);
                double auc = this.getWorkerAUC(worker);
                row += "<td>" + spamScore[0] + "</td>\n";
                row += "<td>" + entropy + "</td>\n";
                row += "<td>" + acc + "</td>\n";
                row += "<td>" + auc + "</td>\n";
                row += "</tr>\n";
                rows += row;
            }
            headerRow += "<td>Worker SS</td>\n";
                headerRow += "<td>Worker Ent.</td>\n";
                headerRow += "<td>Worker Acc.</td>\n";
                headerRow += "<td>Worker Auc.</td>\n";
                footerRow += "<td></td>\n<td></td>\n<td></td>\n<td></td>\n";
            headerRow += "</tr>\n";
            footerRow += "</tr>\n";
            result += headerRow;
            result += rows;
            result += footerRow;
            result += "</table>\n";
            result += "</body>\n";
            result += "</html>";
            return result;
        }
        
}
