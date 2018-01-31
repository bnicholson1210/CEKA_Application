package stat.gaussian;

import java.util.ArrayList;
import java.util.Random;

import stat.StatCalc;
import clustering.KMeans;

public class OneDGaussianMixtureModel extends GaussianMixtureModel
{

	private double[] means;
	private double[] variances;
	
	public OneDGaussianMixtureModel(int numComponents, double[] data)
	{
		double[][] clusterData = new double[data.length][2];
		this.numComponents = numComponents;
		priors = new double[numComponents];
		means = new double[numComponents];
		variances = new double[numComponents];
		ArrayList<ArrayList<Double>> clusterGroups = new ArrayList();
		//Cluster the data, setting the number of initial centroids
		//equal to the number of components
		try
		{
			clusterData = KMeans.performKMeans(numComponents, data);
		}
		catch(Exception e)
		{
			e.printStackTrace();
			System.exit(1);
		}
		//Organize the results into an ArrayList of ArrayLists
		for(int i = 0; i < numComponents; i++)
		{
			clusterGroups.add(new ArrayList<Double>());
		}
		for(int j = 0; j < clusterData.length; j++)
		{
			clusterGroups.get((int)Math.round(clusterData[j][1])).add(clusterData[j][0]);
		}
		//Calculate the initial parameters based on clustering results
		for(int i = 0; i < numComponents; i++)
		{
			priors[i] = (double)clusterGroups.get(i).size() / (double)clusterData.length;
			double[] values = new double[clusterGroups.get(i).size()];
			for(int j = 0; j < clusterGroups.get(i).size(); j++)
			{
				values[j] = clusterGroups.get(i).get(j);
			}
			means[i] = StatCalc.mean(values);
			variances[i] = StatCalc.variance(values);
		}
		//printInfo();
		if(EM(data) == false)
			this.numComponents = -1;
	}
	
	public boolean EM(double[] data)
	{
		boolean result = true;
		ArrayList<Integer> skippables = new ArrayList();
		for(int iterations = 0; iterations < 10; iterations++)
		{
			//Update prior probabilities
			for(int i = 0; i < numComponents; i++)
			{
				for(Integer integer : skippables)
					if(integer == i)
						continue;
				double sum = 0;
				for(int j = 0; j < data.length; j++)
				{
					sum += posteriorProbability(i, data[j], data);
				}
				sum /= (double)data.length;
				priors[i] = sum;
				if(priors[i] == 0)
				{
					result = false;
					//System.out.println("At iteration " + (iterations + 1) + ", prior " + (i + 1) + " is 0.");
					skippables.add(i);
				}
				//System.out.println("PRIOR: " + sum);
			}
			
			//Update Means
			for(int i = 0; i < numComponents; i++)
			{
				for(Integer integer : skippables)
					if(integer == i)
						continue;
				double numSum = 0;
				double denSum = 0;
				for(int j = 0; j < data.length; j++)
				{
					numSum += posteriorProbability(i, data[j], data) * data[j];
					denSum += posteriorProbability(i, data[j], data);
				}
				means[i] = numSum / denSum;
				//System.out.println("MEAN: " + means[i]);
			}
			
			//Update Variances
			for(int i = 0; i < numComponents; i++)
			{
				for(Integer integer : skippables)
					if(integer == i)
						continue;
				double numSum = 0;
				double denSum = 0;
				for(int j = 0; j < data.length; j++)
				{
					//numSum += posteriorProbability(i, data[j], data) * data[j] * data[j];
					numSum += posteriorProbability(i, data[j], data) * Math.pow(data[j] - means[i], 2.0);
					denSum += posteriorProbability(i, data[j], data);
				}
				
				//variances[i] = numSum / denSum - means[i] * means[i];
				variances[i] = numSum / denSum;
				//System.out.println("VARIANCE: " + variances[i]);
			}
		}
		return result;
	}
	
	public double posteriorProbability(int component, double dataPoint, double[] data)
	{

		double interiorSum = 0;
		for(int k = 0; k < numComponents; k++)
		{
			interiorSum += priors[k] * GaussianModel.density(dataPoint, means[k], variances[k]);
		}
		if(new Double(priors[component] * GaussianModel.density(dataPoint, means[component], variances[component]) /
				interiorSum).equals(Double.NaN))
		{
			/*
			System.out.println("The prior of " + component + " is " + priors[component] + 
					", the density function is " + GaussianModel.density(dataPoint, means[component], variances[component])
					+ ", and the total is " + interiorSum);
			for(int k = 0; k < numComponents; k++)
			{
				System.out.println("k: " + k + " prior: " + priors[k] + " mean: " + means[k] + ", variance: " + variances[k] + ", density: " + GaussianModel.density(dataPoint, means[k], variances[k]));
			}
			try
			{
				Thread.sleep(75);
			}
			catch(Exception e)
			{
				System.exit(1);
			}
			*/
		}
		
		return priors[component] * GaussianModel.density(dataPoint, means[component], variances[component]) /
				interiorSum;
		
	}
	
	public void printInfo()
	{
		System.out.println("This GMM has " + numComponents + " components.");
		for(int i = 0; i < numComponents; i++)
		{
			System.out.println("Component " + (i + 1) + ":\n\tMean - " + means[i] +
					"\n\tVariance - " + variances[i] + "\n\tWeight - " + priors[i]);
		}
	}
	
	public double randomNum()
	{
		Random rand = new Random();
		double num = rand.nextDouble();
		int component = -1;
		for(int i = 0; i < priors.length; i++)
		{
			if(num >= priors[i])
				num -= priors[i];
			else
			{
				component = i;
				break;
			}
		}
		if(component == -1)
		{
			//System.out.println("Something wrong with priors.");
			component = priors.length - 1;
		}
		return GaussianModel.randomGaussian(means[component], variances[component]);
	}
	
	public double error(double[] data)
	{
		double outerSum = 0;
		for(int i = 0; i < data.length; i++)
		{
			double innerSum = 0;
			for(int j = 0; j < numComponents; j++)
			{
				System.out.println("For component " + (j + 1) + ": ");
				System.out.println("Density is " + GaussianModel.density(data[i], means[j], variances[j]));
				innerSum += priors[j] * GaussianModel.density(data[i], means[j], variances[j]);
			}
			outerSum += Math.log(innerSum);
		}
		return -1 * outerSum;
	}
	
	//Plots a density plot, with some arbitrarily large number of samples, of this GMM
	public String generateRCode(String datasetName)
	{
		int numbers = 1000000;
		String result = "png('C:\\\\Users\\\\Bryce\\\\Desktop\\\\GMMPlots\\\\" +
			datasetName + "GMMC=" + numComponents + ".png')\n";
		for(int i = 0; i < numComponents; i++)
		{
			double mean = means[i];
			double sd = Math.sqrt(variances[i]);
			int num = (int)(priors[i] * (double)numbers);
			result += "z <- c(z, rnorm(" + num + "," + mean + "," + sd + "))\n";
		}
		result += "plot(density(z), lwd=3, col='red')\n";
		result += "dev.off()\n";
		return result;
	}
}
