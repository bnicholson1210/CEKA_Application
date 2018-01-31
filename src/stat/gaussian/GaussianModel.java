package stat.gaussian;

import java.util.Random;

import stat.StatCalc;
import weka.core.matrix.Matrix;

public class GaussianModel 
{
	double mean;
	double variance;
	
	public GaussianModel(double mean, double variance)
	{
		this.mean = mean;
		this.variance = variance;
	}
	
	public double density(double x)
    {
    	return density(x, mean, variance);
    }
	
	public static double density(double x, double mean, double variance)
	{
		if(variance == 0)
			return 0;
		double result = 1;
    	result = result / (Math.sqrt(variance) * Math.sqrt(2.0 * Math.PI));
    	double a = -1 * Math.pow(x - mean, 2) / (2 * variance);
    	result = result * Math.exp(a);
    	return result;
	}
	
	public static double density(double[] x, double[] mean, double[][] covarianceMatrix)
	{
		//System.out.println("Dim mean: " + mean.length + ", Dim x: " + x.length);
		Matrix covMatrix = new Matrix(covarianceMatrix);
		//System.out.println(covMatrix);
		Matrix inverse = covMatrix.inverse();
		double[][] modMean = new double[mean.length][1];
		double[][] modX = new double[x.length][1];
		for(int i = 0; i < mean.length; i++)
		{
			modMean[i][0] = mean[i];
			modX[i][0] = x[i];
		}
		Matrix meanMatrix = new Matrix(modMean);
		Matrix xMatrix = new Matrix(modX);
		Matrix subMatrix = xMatrix.minus(meanMatrix);
		Matrix subTranspose = subMatrix.copy().transpose();
		double determinant = covMatrix.det();
		int dim = x.length;
		double result = 1;
		result = result / (Math.pow(2 * Math.PI, (double)dim / 2.0) * Math.pow(determinant, 0.5));
		double a = -1 * (subTranspose.times(inverse).times(subMatrix).get(0, 0)) / 2.0;
		result = result * Math.exp(a);
		return result;
	}
	
    public static double randomGaussian(double mean, double variance)
    {
    	Random rand = new Random();
    	return mean + Math.sqrt(variance) * rand.nextGaussian();
    }
    
    public double randomGaussian()
    {
    	return randomGaussian(mean, variance);
    }
}
