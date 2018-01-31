package clustering;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

public class KMeans 
{
	public static Instances createSingleDimData(double[] data) throws Exception
	{
		File f = new File("singeDimData.arff");
		f.delete();
		BufferedWriter bw = new BufferedWriter(new FileWriter(f));
		bw.write("@relation singleDimData\n");
		bw.write("@attribute att1 real\n");
		bw.write("@data\n");
		for(int i = 0; i < data.length; i++)
		{
			bw.write("" + data[i] + "\n");
		}
		bw.close();
		FileReader reader = new FileReader(f.getAbsolutePath());
		BufferedReader readerBuffer = new BufferedReader(reader);
		Instances instSet = new Instances(readerBuffer);
		instSet.setClassIndex(-1);
		return instSet;
	}
	
	public static Instances createMultiDimData(double[][] data) throws Exception
	{
		int dim = data[0].length;
		File f = new File("multiDimData.arff");
		f.delete();
		BufferedWriter bw = new BufferedWriter(new FileWriter(f));
		bw.write("@relation multiDimData\n");
		for(int i = 0; i < dim; i++)
		{
			bw.write("@attribute att" + i + " real\n");
		}
		bw.write("@data\n");
		for(int i = 0; i < data.length; i++)
		{
			for(int j = 0; j < data[i].length; j++)
			{
				bw.write("" + data[i][j]);
				if(j != data[i].length - 1)
					bw.write(",");
			}
			bw.write("\n");
		}
		bw.close();
		FileReader reader = new FileReader(f.getAbsolutePath());
		BufferedReader readerBuffer = new BufferedReader(reader);
		Instances instSet = new Instances(readerBuffer);
		instSet.setClassIndex(-1);
		return instSet;
	}
	
	//For one-dimensional data
	public static double[][] performKMeans(int numCentroids, double[] dataPoints) throws Exception
	{ 
		//Save the result of the clustering in the form of two arrays,
		//where the first array saves the data points themselves and
		//the second array saves the cluster number that each respective
		//data point is assigned to.
		boolean goAgain = true;
		double[][] result = new double[1][1];
		while(goAgain)
		{
			goAgain = false;
			Instances data = createSingleDimData(dataPoints);
			result = new double[data.numInstances()][2];
			SimpleKMeans clusterer = new SimpleKMeans();
			clusterer.setNumClusters(numCentroids);
			clusterer.setSeed((int)(System.nanoTime() % 1000000));
			clusterer.buildClusterer(data);
			int[] counters = new int[numCentroids];
			if((double)data.numInstances() / (double)numCentroids < 2)
			{
				System.out.println("Data cannot handle this many centroids!");
				System.exit(1);
			}
			for(int i = 0; i < data.numInstances(); i++)
			{
				result[i][1] = clusterer.clusterInstance(data.instance(i));
				result[i][0] = data.instance(i).value(0);
				counters[(int)Math.round(result[i][1])]++;
				//System.out.println("" + result[i][0] + "\t" + result[i][1]);
			}
			for(int i = 0; i < numCentroids; i++)
			{
				if(counters[i] < 2)
				{
					//System.out.println("i: " + i + ", counters[i]: " + counters[i]);
					goAgain = true;
					break;
				}
			}
		}
		return result;
	}
	
	//For multi-dimensional data
	//E.G., for the workers in the crowdsourcing model, where the worker has attributes:
	//{class0tendency0, class1tendency0, numLabels}
	public static double[][] performKMeansMD(int numCentroids, double[][] dataPoints) throws Exception
	{ 
		//Save the result of the clustering in the form of two arrays,
		//where the first array saves the data points themselves and
		//the second array saves the cluster number that each respective
		//data point is assigned to.
		boolean goAgain = true;
		double[][] result = new double[1][1];
		while(goAgain)
		{
			goAgain = false;
			Instances data = createMultiDimData(dataPoints);
			int numData = dataPoints.length;
			int dimension = dataPoints[0].length;
			result = new double[numData][dimension + 1];
			SimpleKMeans clusterer = new SimpleKMeans();
			clusterer.setNumClusters(numCentroids);
			clusterer.setSeed((int)(System.nanoTime() % 1000000));
			clusterer.buildClusterer(data);
			int[] counters = new int[numCentroids];
			if((double)data.numInstances() / (double)numCentroids < 2)
			{
				System.out.println("Data cannot handle this many centroids!");
				System.exit(1);
			}
			for(int i = 0; i < data.numInstances(); i++)
			{
				result[i][dimension] = clusterer.clusterInstance(data.instance(i));
				for(int j = 0; j < dimension; j++)
				{
					result[i][j] = data.instance(i).value(j);
				}
				counters[(int)Math.round(result[i][dimension])]++;
				//System.out.println("" + result[i][0] + "\t" + result[i][1]);
			}
			for(int i = 0; i < numCentroids; i++)
			{
				if(counters[i] < 2)
				{
					//System.out.println("i: " + i + ", counters[i]: " + counters[i]);
					goAgain = true;
					break;
				}
			}
		}
		return result;
	}
}
