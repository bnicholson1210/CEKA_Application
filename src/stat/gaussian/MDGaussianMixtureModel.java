package stat.gaussian;

import java.util.ArrayList;
import java.util.Random;

import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;
import stat.StatCalc;
import clustering.KMeans;

public class MDGaussianMixtureModel extends GaussianMixtureModel
{
	double[][] means;
	double[][][] covarianceMatrices;
	public MDGaussianMixtureModel(int numComponents, double[][] data)
	{
		data = StatCalc.filterVectors(data, true);
		int dim = data[0].length;
		double[][] clusterData = new double[data.length][2];
		this.numComponents = numComponents;
		priors = new double[numComponents];
		means = new double[numComponents][dim];
		covarianceMatrices = new double[numComponents][dim][dim];
		//variances = new double[numComponents][dim];
		ArrayList<ArrayList<ArrayList<Double>>> clusterGroups = new ArrayList();
		//Cluster the data, setting the number of initial centroids
		//equal to the number of components
		try
		{
			clusterData = KMeans.performKMeansMD(numComponents, data);
		}
		catch(Exception e)
		{
			e.printStackTrace();
			System.exit(1);
		}
		
		//Organize the results into an ArrayList of ArrayLists
		for(int i = 0; i < numComponents; i++)
		{
			clusterGroups.add(new ArrayList<ArrayList<Double>>());
		}
		for(int j = 0; j < clusterData.length; j++)
		{
			ArrayList<Double> list = new ArrayList();
			for(int k = 0; k < dim; k++)
			{
				list.add(clusterData[j][k]);
			}
			clusterGroups.get((int)Math.round(clusterData[j][dim])).add(list);
		}
		//Calculate the initial parameters based on clustering results
		for(int i = 0; i < numComponents; i++)
		{
			priors[i] = (double)clusterGroups.get(i).size() / (double)clusterData.length;
			double[][] values = new double[clusterGroups.get(i).size()][dim];
			for(int j = 0; j < clusterGroups.get(i).size(); j++)
			{
				for(int k = 0; k < clusterGroups.get(i).get(j).size(); k++)
				{
					values[j][k] = clusterGroups.get(i).get(j).get(k);
				}
			}
			means[i] = StatCalc.mean(values);
			covarianceMatrices[i] = StatCalc.calculateCovarianceMatrix(values);
		}
		//printInfo();
		try
		{
			if(EM(data) == false)
				this.numComponents = -1;
		}
		catch(java.lang.RuntimeException re)
		{
			this.numComponents = -1;
			return;
		}
	}
	
	public boolean EM(double[][] data) throws java.lang.RuntimeException
	{
		boolean result = true;
		int dim = data[0].length;
		for(int iterations = 0; iterations < 5; iterations++)
		{
			//Update prior probabilities
			for(int i = 0; i < numComponents; i++)
			{
				double sum = 0;
				for(int j = 0; j < data.length; j++)
				{
					sum += posteriorProbability(i, data[j], data);
				}
				sum /= (double)data.length;
				priors[i] = sum;
				if(priors[i] == 0 || !(priors[i] > Double.NEGATIVE_INFINITY && priors[i] < Double.POSITIVE_INFINITY))
				{
					result = false;
					//System.out.println("At iteration " + (iterations + 1) + ", prior " + (i + 1) + " is invalid.");
				}
				//System.out.println("PRIOR: " + sum);
			}
			
			//Update Means
			for(int i = 0; i < numComponents; i++)
			{
				double[] sum = new double[dim];
				for(int j = 0; j < data.length; j++)
				{
					sum = StatCalc.addArrays(sum, StatCalc.scaleArray(posteriorProbability(i, data[j], data), data[j]));
				}
				double denSum = 0;
				for(int j = 0; j < data.length; j++)
				{
					denSum += posteriorProbability(i, data[j], data);
				}
				sum = StatCalc.scaleArray(1 / denSum, sum);
				means[i] = sum;
				//System.out.println("MEAN: " + means[i]);
			}
			
			//Update Variances
			for(int i = 0; i < numComponents; i++)
			{
				double numSum = 0;
				double scale = 0;
				Matrix sum = new Matrix(new double[dim][dim]);
				double otherSum = 0;
				for(int j = 0; j < data.length; j++)
				{
					double[] subArray = StatCalc.subtractArrays(data[j], means[i]);
					Matrix x = new Matrix(StatCalc.addDimension(subArray));
					Matrix xTrans = x.transpose();
					sum = sum.plus(x.times(xTrans).times(posteriorProbability(i, data[j], data)));
				    otherSum = otherSum + posteriorProbability(i, data[j], data);
				}
				sum = sum.times(1.0 / otherSum);
				covarianceMatrices[i] = sum.getArray();
				for(int j = 0; j < dim; j++)
				{
					for(int k = 0; k < dim; k++)
					{
						if(!(covarianceMatrices[i][j][k] > Double.NEGATIVE_INFINITY && 
								covarianceMatrices[i][j][k] < Double.POSITIVE_INFINITY))
							this.numComponents = -1;
					}
				}
			}
			//System.out.println("Iteration " + iterations + " done.");
			//printInfo();
		}
		
		try
		{
			for(int i = 0; i < numComponents; i++)
			{
				Matrix a = new Matrix(covarianceMatrices[i]);
				Matrix b = a.inverse();
			}
		}
		catch(java.lang.RuntimeException re)
		{
			result = false;
		}
		//System.out.println("Exiting EM function, " + result + ".");
		return result;
	}
	
	public double posteriorProbability(int component, double[] dataPoint, double[][] data)
	{
		double interiorSum = 0;
		for(int k = 0; k < numComponents; k++)
		{
			interiorSum += priors[k] * GaussianModel.density(dataPoint, means[k], covarianceMatrices[k]);
		}
		if(new Double(priors[component] * GaussianModel.density(dataPoint, means[component], covarianceMatrices[component]) /
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
		
		return priors[component] * GaussianModel.density(dataPoint, means[component], covarianceMatrices[component]) /
				interiorSum;
		
	}
	
	public double[] randomVector()
	{
		int dim = means[0].length;
		Random rand = new Random();
		double num = rand.nextDouble();
		//Create random vector of N(0, I)
		double[] x = new double[dim];
		for(int i = 0; i < dim; i++)
		{
			x[i] = rand.nextGaussian();
		}
		//Choose component randomly from which to take sample
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
		Matrix cov = new Matrix(covarianceMatrices[component]);
		EigenvalueDecomposition eig = cov.eig();
		//phi, the eigenvector-columns matrix
		Matrix phi = StatCalc.normalizeColumns(eig.getV());
		//lambda, the diagonal eigenvalues matrix
		Matrix lambda = eig.getD();
		//Q
		Matrix Q = lambda.sqrt().times(phi);
		Matrix vectorX = new Matrix(StatCalc.addDimension(x));
		//vectorX = vectorX.transpose();
		Matrix meanMatrix = new Matrix(StatCalc.addDimension(means[component]));
		//meanMatrix = meanMatrix.transpose();
		Matrix newVector = Q.times(vectorX).plus(meanMatrix);
		newVector = newVector.transpose();
		double[][] newArray = newVector.getArray();
		double[] result = new double[newArray[0].length];
		for(int i = 0; i < newArray[0].length; i++)
		{
			result[i] = newArray[0][i];
		}
		return result;
	}
	
	public void printInfo()
	{
		String message = "This GMM has " + numComponents + " components.\n";
		for(int i = 0; i < numComponents; i++)
		{
			message += "For component " + (i + 1) + ": \n";
			message += "\tPrior - " + priors[i] + "\n";
			message += "\tMean - {";
			for(int j = 0; j < means[i].length; j++)
			{
				message += means[i][j];
				if(j != means[i].length - 1)
					message += ", ";
			}
			message += "}\n\tCovariance Matrix - \n" + new Matrix(covarianceMatrices[i]).toString() + "\n";
		}
		System.out.println(message);
	}
}
