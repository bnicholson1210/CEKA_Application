package stat;

import ceka.utils.Misc;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import stat.gaussian.GaussianModel;
import weka.core.matrix.Matrix;

public class StatCalc 
{
    
    public static double mode(double[] values){
        Map<Double, Integer> valueCounts = new HashMap<>();
        for(int i = 0; i < values.length; i++){
            Integer prevCount = valueCounts.get(values[i]);
            if(prevCount == null){
                prevCount = 0;
            }
            valueCounts.put(values[i], prevCount + 1);
        }
        double maxValue = 0;
        Integer maxCount = Integer.MIN_VALUE;
        Iterator<Double> it = valueCounts.keySet().iterator();
        while(it.hasNext()){
            Double key = it.next();
            if(valueCounts.get(key) > maxCount){
                maxCount = valueCounts.get(key);
                maxValue = key;
            }
        }
        return maxValue;
    }
    
    public static Double mode(List<Double> values){
        double[] primitiveValues = new double[values.size()];
        for(int i = 0; i < values.size(); i++){
            primitiveValues[i] = values.get(i);
        }
        return mode(primitiveValues);
    }
    
	public static double mean(double[] values)
    {
        double sum = 0;
        int numUndefined = 0;
        for(int i = 0; i < values.length; i++)
        {
        	if(values[i] > Double.NEGATIVE_INFINITY && values[i] < Double.POSITIVE_INFINITY)
        		sum += values[i];
                else
        		numUndefined++;
        }
        sum /= (double)(values.length - numUndefined);
        return sum;
    }
	
	public static double[] mean(double[][] values)
	{
		double[] result = new double[values[0].length];
		for(int i = 0; i < values[0].length; i++)
		{
			double[] array = new double[values.length];
			for(int j = 0; j < values.length; j++)
			{
				array[j] = values[j][i];
			}
			result[i] = mean(array);
		}
		return result;
	}
        
    public static double mean(List<Double> values)
    {
        double[] vals = new double[values.size()];
        for(int i = 0; i < values.size(); i++)
        {
            vals[i] = values.get(i).doubleValue();
        }
        return  mean(vals);
    }
    
    public static double variance(double[] values)
    {
        double mean = mean(values);
        double[] squareDifferences = new double[values.length];
        for(int i = 0; i < values.length; i++)
        {
            squareDifferences[i] = Math.pow(values[i] - mean, 2.0);
        }
        return mean(squareDifferences);
    }  
    
    public static double variance(ArrayList<Double> values)
    {
        double[] vals = new double[values.size()];
        for(int i = 0; i < values.size(); i++)
        {
            vals[i] = values.get(i).doubleValue();
        }
        return  variance(vals);
    }
    
    public static double[] variance(double[][] values)
    {
    	double[] result = new double[values[0].length];
		for(int i = 0; i < values[0].length; i++)
		{
			double[] array = new double[values.length];
			for(int j = 0; j < values.length; j++)
			{
				array[j] = values[j][i];
			}
			result[i] = variance(array);
		}
		return result;
    }
    
    public static double[] standardDeviation(double[][] values)
    {
        double[] variance = variance(values);
        for(int i = 0; i < variance.length; i++)
        {
            variance[i] = Math.sqrt(variance[i]);
        }
        return variance;
    }
    
    public static double cosineSimilarity(double[] arr1, double[] arr2){
        if(arr1.length != arr2.length) throw new Error("Cannot calculate cosine similarity on arrays of different lengths."); 
        double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
        for(int i = 0; i < arr1.length; i++) {
           dot += arr1[i] * arr2[i] ;
           denom_a += arr1[i] * arr1[i] ;
           denom_b += arr2[i] * arr2[i] ;
       }
       double result = dot / (Math.sqrt(denom_a) * Math.sqrt(denom_b));
       if(Double.isNaN(result)) throw new Error("The cosine similarity was NaN!");
       else return result;
       //if(Double.isNaN(result)) System.out.println("The cosine similarity was NaN!");
       //return result;
    }
    
    public static double[] filter(double[] values, boolean randomize)
    {
    	int counter = 0;
    	double[] filteredValues;
    	for(int i = 0; i < values.length; i++)
    	{
    		if(values[i] < Double.POSITIVE_INFINITY && values[i] > Double.NEGATIVE_INFINITY)
    			counter++;
    	}
    	filteredValues = new double[counter];
    	counter = 0;
    	for(int i = 0; i < values.length; i++)
    	{
    		if(values[i] < Double.POSITIVE_INFINITY && values[i] > Double.NEGATIVE_INFINITY)
    		{
    			filteredValues[counter] = values[i];
    			counter++;
    		}
    	}
    	//For the purpose of eliminating duplicate values in the data
    	if(randomize)
    	{
	    	GaussianModel noiseModel = new GaussianModel(0, .00000005);
	    	for(int i = 0; i < filteredValues.length; i++)
	    	{
	    		filteredValues[i] += noiseModel.randomGaussian();
	    		//System.out.print(filteredValues[i] + " ");
	    	}
    	}
    	//System.out.println();
    	return filteredValues;
    }
    
    //Plots the density of the data points given
    public static String generateRCode(double[] data, String title)
    {
    	String result = "png('C:\\\\Users\\\\Nick\\\\Desktop\\\\GMMPlots\\\\" +
    			title + ".png')\n";
    	result += "z <- c(";
    	for(int i = 0; i < data.length; i++)
    	{
    		result += "" + data[i];
    		if(i != data.length - 1)
    			result += ",";
    	}
    	result += ")\n";
    	result += "plot(density(z), lwd=3, col='red')\n";
		result += "dev.off()\n";
		return result;
    }
    
    public static double[][] calculateCovarianceMatrix(double[][] values)
    {
    	int dim = values[0].length;
    	double[][] result = new double[dim][dim];
    	double[] means = new double[dim];
    	for(int i = 0; i < dim; i++)
    	{
    		double[] array = new double[values.length];
    		for(int j = 0; j < values.length; j++)
    		{
    			array[j] = values[j][i];
    		}
    		means[i] = mean(array);
    		//System.out.println("mean " + i + ": " + means[i]);
    	}
    	for(int i = 0; i < dim; i++)
    	{
    		for(int j = 0; j < dim; j++)
    		{
    			double sum = 0;
    			for(int k = 0; k < values.length; k++)
    			{
    				//System.out.println("i = " + i + "\nj = " + j + "\nk = "
    						//+ k + "\nvalue k,i: " + values[k][i] + "\nvalue k,j: " + values[k][j]);
    				sum += (values[k][i] - means[i]) * (values[k][j] - means[j]);
    			}
    			sum /= (double)values.length;
    			result[i][j] = sum;
    		}
    	}
    	return result;

    }
    
    public static double[] scaleArray(double scale, double[] values)
    {
    	double[] result = new double[values.length];
    	for(int i = 0; i < values.length; i++)
    	{
    		result[i] = values[i] * scale;
    	}
    	return result;
    }
    
    public static double[] subtractArrays(double[] array1, double[] array2)
    {
    	double[] result = new double[array1.length];
    	for(int i = 0; i < result.length; i++)
    	{
    		result[i] = array1[i] - array2[i];
    	}
    	return result;
    }
    
    public static double[] addArrays(double[] array1, double[] array2)
    {
    	double[] result = new double[array1.length];
    	for(int i = 0; i < result.length; i++)
    	{
    		result[i] = array1[i] + array2[i];
    	}
    	return result;
    }
    
    public static double[][] addDimension(double[] array)
    {
    	double[][] result = new double[array.length][1];
    	for(int i = 0; i < array.length; i++)
    	{
    		result[i][0] = array[i];
    	}
    	return result;
    }
    
    public static Matrix normalizeColumns(Matrix a)
    {
    	double[][] matrix = a.getArray();
    	for(int i = 0; i < matrix[0].length; i++)
    	{
    		double sum = 0;
    		for(int j = 0; j < matrix.length; j++)
    		{
    			sum += Math.pow(matrix[j][i], 2.0);
    		}
    		sum = Math.sqrt(sum);
    		for(int j = 0; j < matrix.length; j++)
    		{
    			matrix[j][i] /= sum;
    		}
    	}
    	return new Matrix(matrix);
    }
    
    public static double[][] filterVectors(double[][] vectors, boolean randomize)
    {
    	int counter = 0;
    	for(int i = 0; i < vectors.length; i++)
    	{
    		boolean nan = false;
    		for(int j = 0; j < vectors[i].length; j++)
    		{
    			double num = vectors[i][j];
    			if(!(num > Double.NEGATIVE_INFINITY && num < Double.POSITIVE_INFINITY))
    				nan = true;
    		}
    		if(!nan)
    			counter++;
    	}
    	double[][] result = new double[counter][vectors[0].length];
    	counter = 0;
    	for(int i = 0; i < vectors.length; i++)
    	{
    		boolean nan = false;
    		for(int j = 0; j < vectors[i].length; j++)
    		{
    			double num = vectors[i][j];
    			if(!(num > Double.NEGATIVE_INFINITY && num < Double.POSITIVE_INFINITY))
    				nan = true;
    		}
    		if(!nan)
    		{
    			for(int j = 0; j < vectors[i].length; j++)
    			{
    				result[counter][j] = vectors[i][j];
    			}
   				counter++;
    		}
    	}
    	if(randomize)
    	{
    		GaussianModel noiseModel = new GaussianModel(0, .000005);
    		for(int i = 0; i < result.length; i++)
    		{
    			for(int j = 0; j < result[0].length; j++)
    			{
    				result[i][j] += noiseModel.randomGaussian();
    			}
    		}
    	}
    	return result;
    }
    public static BigDecimal choose(double n, double k)
    {
    	BigDecimal num = new BigDecimal(1.0);
    	BigDecimal denom = new BigDecimal(1.0);
    	for(double i = 0; i < k; i++)
    	{
                num = num.multiply(new BigDecimal(n));
                denom = denom.multiply(new BigDecimal((double)(i + 1)));
                n = n + 1;
    	}
    	return num.divide(denom);
    }
    
    public static int factorial(int n)
    {
    	int prod = 1;
    	for(int i = n; i > 0; i--)
    	{
    		prod = prod * n;
    	}
    	return prod;
    }
    
    public static double[] normalize(double[] values)
    {
    	double sum = 0;
    	for(int i = 0; i < values.length; i++)
    	{
    		sum += values[i];
    	}
    	for(int i = 0; i < values.length; i++)
    	{
    		values[i] /= sum;
    	}
    	return values;
    }
    
    public static double[] normalize(int[] values)
    {
    	double[] vals = new double[values.length];
    	for(int i = 0; i < vals.length; i++)
    	{
    		vals[i] = (double)values[i];
    	}
    	return normalize(vals);
    }
    
    public static double[] removeZeros(double[] values)
    {
        int counter = 0;
        for(int i = 0; i < values.length; i++)
        {
            if(values[i] != 0)
                counter++;
        }
        double[] result = new double[counter];
        counter = 0;
        for(int i = 0; i < values.length; i++)
        {
            if(values[i] != 0)
            {
                result[counter] = values[i];
                counter++;
            }
        }
        return result;
    }
    
    public static double[][] removeZeros(double[][] values)
    {
    	int numKeepRows = 0;
    	for(int i = 0; i < values.length; i++)
    	{
    		boolean removeRow = true;
    		for(int j = 0; j < values[i].length; j++)
    		{
    			if(values[i][j] != 0)
    			{
    				removeRow = false;
    				break;
    			}	
    		}
    		if(removeRow == false)
    			numKeepRows++;
    	}
    	double[][] result = new double[numKeepRows][values[0].length];
    	int rowCounter = 0;
    	for(int i = 0; i < values.length; i++)
    	{
    		boolean removeRow = true;
    		for(int j = 0; j < values[i].length; j++)
    		{
    			if(values[i][j] != 0)
    			{
    				removeRow = false;
    				break;
    			}	
    		}
    		if(removeRow == false)
    		{
    			for(int j = 0; j < values[i].length; j++)
    			{
    				result[rowCounter][j] = values[i][j];
    			}
    			rowCounter++;
    		}
    	}
    	
    	return result;
    }
    public static double[] quartiles(double[] data)
    {
        if(data.length == 0) return new double[]{Double.NaN, Double.NaN, Double.NaN, Double.NaN};
        Arrays.sort(data);
        double length = data.length;
        double n1 = 1.0 / 4.0 * (length + 1);
        double q1 = data[(int)n1]; //* (1.0 + (n1 - (int)n1));
        double n2 = 2.0 / 4.0 * (length + 1);
        double q2 = data[(int)n2];// * (1.0 + (n2 - (int)n2));
        double n3 = 3.0 / 4.0 * (length + 1);
        double q3 = data[(int)n3];// * (1 + (n3 - (int)n3));
        double q4 = data[data.length - 1];
        double[] quartiles = {q1, q2, q3, q4};
        return quartiles;
    }
    
    public static void printArray(double[] arr)
    {
        for(int i = 0; i < arr.length; i++)
        {
            System.out.print(arr[i]);
            if(i != arr.length - 1)
                System.out.print(",");
        }
        System.out.println();
    }
    
    public static int maxIndex(double[] arr)
    {
        double max = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;
        for(int i = 0; i < arr.length; i++)
        {
            if(arr[i] > max)
            {
                max = arr[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    public static double findPercentile(double[] data, double percentile)
    {
        Arrays.sort(data);
        int index = (int)(percentile * (double)data.length);
        return data[index];
    }
    
    public static Double findPercentileOfPoint(ArrayList<Double> data, Double point)
    {
        int size = data.size();
        ArrayList<Double> copy = new ArrayList();
        for(Double d : data)
        {
            copy.add(d);
        }
        Collections.sort(copy);
        int counter = 0;
        for(int i = 0; i < copy.size(); i++)
        {
            if(point < copy.get(i))
                break;
            counter++;
        }
        Double result = (double)counter / (double)size;
        return result;
    }
    
    public static double[] arrayCopy(double[] array)
    {
        double[] copy = new double[array.length];
        for(int i = 0; i < array.length; i++)
        {
            copy[i] = array[i];
        }
        return copy;
    }
    
    public static double min(double[] array)
    {
        double min = Double.POSITIVE_INFINITY;
        for(int i = 0; i < array.length; i++)
        {
            if(array[i] < min)
                min = array[i];
        }
        return min;
    }
    
    public static double max(double[] array)
    {
        double max = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < array.length; i++)
        {
            if(array[i] > max)
                max = array[i];
        }
        return max;
    }
    
    public static double pearsonsR(double[] vals1, double[] vals2)
    {
        double mean1 = mean(vals1);
        double mean2 = mean(vals2);
        
        double covSum = 0;
        for(int i = 0; i < vals1.length; i++)
        {
            covSum += (vals1[i] - mean1)*(vals2[i] - mean2);
        }
        covSum /= ((double)vals1.length - 1.0);
        covSum /= Math.sqrt(variance(vals1));
        covSum /= Math.sqrt(variance(vals2));
        return covSum;
    }
    
    public static double calculateAUC(int c1, int c2, int [] predictedLabels,  int [] realLabels, boolean convex) {
		
		double auc = 0;
		// find all example with realLabel= C1 && (predictedLabels = C1 or C2);
		ArrayList<Integer> realC1 = new ArrayList<Integer>();
		ArrayList<Double>  predC12 = new ArrayList<Double>();
		
		for (int i = 0; i < realLabels.length; i++) {
			if ((realLabels[i] == c1) && ((predictedLabels[i] == c1) ||  (predictedLabels[i] == c2))) {
				realC1.add(realLabels[i]);
				predC12.add(new Double(predictedLabels[i]));
			}
			if ((realLabels[i] == c2) && ((predictedLabels[i] == c1) ||  (predictedLabels[i] == c2))) {
				realC1.add(realLabels[i]);
				predC12.add(new Double(predictedLabels[i]));
			}
		}
		
		if (realC1.size() > 0) {
			int [] real = new int[realC1.size()];
			double [] pred = new double [realC1.size()];
			for (int i = 0; i < realC1.size(); i++) {
				if (realC1.get(i).intValue() == c2)
					real[i] = 1;
				else
					real[i] = 0;
				if (Misc.isDoubleSame(predC12.get(i).doubleValue(), (double)c2, 0.0000001))
					pred[i] = 1.0;
				else
					pred[i] = 0.0;
			}
			mloss.roc.Curve rocAnalysis = new mloss.roc.Curve.PrimitivesBuilder().predicteds(pred).actuals(real).build();
			if (convex) {
				// Get the convex hull
			    mloss.roc.Curve convexHull = rocAnalysis.convexHull();
			    auc = convexHull.rocArea();
			    if (Double.isNaN(auc))
			    	auc = 0;
			    //log.debug("AUC_Convex ("+c1 + "," + c2 +")=" + auc);
			} else {
				auc = rocAnalysis.rocArea();
				if (Double.isNaN(auc))
				    auc = 0;
				//log.debug("AUC ("+c1 + "," + c2 +")=" + auc + "    ");
			}
		}
		
		return auc;
	}
    
    public static int sumOfElements(double[][] matrix)
    {
        int total = 0;
        for(int i = 0; i < matrix.length; i++)
        {
            for(int j = 0; j < matrix[i].length; j++)
            {
                total += (int)matrix[i][j];
            }
        }
        return total;
    }
    
    public static double getAUC(double[][] confusionMatrix)
    {
        int dim = confusionMatrix.length;
        int total = (int)sumOfElements(confusionMatrix);
        int[] predLabels = new int[total];
        int[] trueLabels = new int[total];
        int index = 0;
        double auc = 0;
        int numCategory = confusionMatrix.length;
        for(int i = 0; i < confusionMatrix.length; i++)
        {
            for(int j = 0; j < confusionMatrix.length; j++)
            {
                int elem = (int)confusionMatrix[i][j];
                for(int k = 0; k < elem; k++)
                {
                    predLabels[index] = j;
                    trueLabels[index] = i;
                    index++;
                }
            }
        }
        for (int i = 0; i < numCategory - 1; i++)	
        {
                for (int j = i + 1; j < numCategory; j++)
                {
                        auc += calculateAUC(i, j, predLabels, trueLabels, false);
                }
        }
       auc = (2 * auc) / (double) (numCategory * (numCategory - 1));
       return auc;
    }
    
    public static double[] scaleArray(double[] array, double scalar)
    {
        double[] result = new double[array.length];
        for(int i = 0; i < array.length; i++)
        {
            result[i] = scalar * array[i];
        }
        return result;
    }
    
    public static ArrayList<Double> getFirstHalfArrayList(double[] array)
    {
        ArrayList<Double> result = new ArrayList();
        for(int i = 0; i < 9; i++)
        {
            result.add(array[i]);
        }
        return result;
    }
    
    public static ArrayList<Double> getSecondHalfArrayList(double[] array)
    {
        ArrayList<Double> result = new ArrayList();
        for(int i = 9; i < 18; i++)
        {
            result.add(array[i]);
        }
        return result;
    }
    
    public static HashMap<String, Double> getRelativeDistribution(HashMap<String, Integer> totalLabelCounts, HashMap<String, Integer> individualLabelCounts){
        HashMap<String, Double> totalProportion = new HashMap();
        HashMap<String, Double> individualProportion = new HashMap();
        Iterator<String> it = totalLabelCounts.keySet().iterator();
        Integer sum = 0;
        while(it.hasNext()){
            String key = it.next();
            Integer count = totalLabelCounts.get(key);
            sum += count;
        }
        it = totalLabelCounts.keySet().iterator();
        while(it.hasNext()){
            String key = it.next();
            Integer count = totalLabelCounts.get(key);
            Double value = (double)count / (double)sum;
            totalProportion.put(key, value);
        }
        
        Iterator<String> individualIt = individualLabelCounts.keySet().iterator();
        Integer individualSum = 0;
        while(individualIt.hasNext()){
            String key = individualIt.next();
            Integer count = individualLabelCounts.get(key);
            individualSum += count;
        }
        individualIt = individualLabelCounts.keySet().iterator();
        while(individualIt.hasNext()){
            String key = individualIt.next();
            Integer count = individualLabelCounts.get(key);
            Double value = (double)count / (double)individualSum;
            individualProportion.put(key, value);
        }
        HashMap<String, Double> result = new HashMap();
        Iterator<String> overallIt = totalLabelCounts.keySet().iterator();
        while(overallIt.hasNext()){
            String key = overallIt.next();
            Double individualProp = individualProportion.get(key);
            if(individualProp == null) individualProp = 0.0;
            Double totalProp = totalProportion.get(key);
            if(totalProp == null) totalProp = 0.0;
            result.put(key, (double)individualProp * (double)individualProp / (double)totalProportion.get(key));
        }
        return normalizeHashMap(result);
    }
    
    public static HashMap<String, Double> normalizeHashMap(HashMap<String, Double> hashMap){
        Iterator<String> it = hashMap.keySet().iterator();
        Double sum = 0.0;
        while(it.hasNext()){
            String key = it.next();
            sum += hashMap.get(key);
        }
        it = hashMap.keySet().iterator();
        while(it.hasNext()){
            String key = it.next();
            if(sum == 0) hashMap.put(key, 0.0);
            else hashMap.put(key, hashMap.get(key) / sum);
        }
        return hashMap;
    }
    
    public static double entropy(HashMap<String, Double> hashMap){
        Iterator<String> it = hashMap.keySet().iterator();
        double entropy = 0;
        while(it.hasNext()){
            String key = it.next();
            Double value = hashMap.get(key);
            if(value != 0)
                entropy += -1 * value * Math.log(value);
        }
        return entropy;
    }
}
