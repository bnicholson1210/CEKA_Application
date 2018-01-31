package research.crowdsourcing;

import weka.core.Instance;
import ceka.core.Example;

public class AnalyzedTask extends Example implements Comparable<AnalyzedTask>
{
	private String id;
	protected double[] tendencies;
	private int numLabels;
	public AnalyzedTask(Example e, int numClasses)
	{
		super((Instance)e);
		this.setTrueLabel(e.getTrueLabel());
		id = e.getId();
		tendencies = new double[numClasses];
		numLabels = 0;
	}
	
	public String getId()
	{
		return id;
	}
	
	public String toString()
	{
		String result = "Task " + id + "\n" + getTrueLabel().getValue() + " |";
		int dimension = tendencies.length;
		for(int i = 0; i < dimension; i++)
		{
			result += tendencies[i];
			if(i != dimension - 1)
				result += "\t";
		}
		result += "|\n";
		return result;
	}
	
	public void updateNumLabels()
	{
		numLabels = this.getMultipleNoisyLabelSet(0).getLabelSetSize();
		return;
	}
	
	public int compareTo(AnalyzedTask other)
	{
		if(this.numLabels > other.numLabels)
			return 1;
		else if(this.numLabels < other.numLabels)
			return -1;
		else
			return 0;
	}
}
