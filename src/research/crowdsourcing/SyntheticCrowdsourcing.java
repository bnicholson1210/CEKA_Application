package research.crowdsourcing;

import ceka.consensus.ds.DawidSkene;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import stat.StatCalc;
import stat.gaussian.GaussianMixtureModel;
import stat.gaussian.MDGaussianMixtureModel;
import stat.gaussian.OneDGaussianMixtureModel;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;

public class SyntheticCrowdsourcing 
{
	public static final int MAX_GUESSES = 50;
	
	public SyntheticCrowdsourcing()
	{
	
	}
	
	public void simulateCrowdsourcing(Dataset dataset)
	{
	
	}
	
	public static ArrayList<AnalyzedTask> getTasksForDataset(Dataset dataset)
	{
		ArrayList<AnalyzedTask> tasks = new ArrayList();
		int numCategories = dataset.getCategorySize();
		for(int i = 0; i < dataset.getExampleSize(); i++)
		{
                    AnalyzedTask t = new AnalyzedTask(dataset.getExampleByIndex(i), numCategories);
                    t.setIntegratedLabel(dataset.getExampleByIndex(i).getIntegratedLabel());
                    tasks.add(t);
		}
		return tasks;
	}
	
	public static ArrayList<AnalyzedWorker> getWorkersForDataset(Dataset dataset)
	{
		ArrayList<AnalyzedWorker> workers = new ArrayList();
		int numWorkers = dataset.getWorkerSize();
		int numCategories = dataset.getCategorySize();
		for(int i = 0; i < numWorkers; i++)
		{
			workers.add(new AnalyzedWorker(dataset.getWorkerByIndex(i), numCategories, true));
		}
		return workers;
	}      
}
