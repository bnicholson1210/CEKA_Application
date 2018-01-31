package stat.gaussian;

import java.util.ArrayList;
import java.util.Random;

import stat.StatCalc;
import clustering.KMeans;

public abstract class GaussianMixtureModel 
{
	protected int numComponents;
	protected double[] priors;
	
	
	public boolean isValid()
	{
		return(numComponents != -1);
	}
}
