package research.crowdsourcing;

import ceka.core.Worker;

public class AnalyzedWorker extends Worker
{
	protected double[][] tendencies;
	protected String id;
        public Worker worker;
        public double sim = 0;
	int numToLabel = 0;

	public AnalyzedWorker(Worker w, int numClasses, boolean includeLabels)
	{
		super("");
                worker = w;
		tendencies = new double[numClasses][numClasses];
		id = w.getId();
		if(includeLabels)
		{
			int numLabels = w.getMultipleNoisyLabelSet(0).getLabelSetSize();
			for(int i = 0; i < numLabels; i++)
			{
				addNoisyLabel(w.getMultipleNoisyLabelSet(0).getLabel(i));
			}
		}
	}
	
	public String getId()
	{
		return id;
	}
	
	public String toString()
	{
		String result = "Worker " + id + "\n";
		int dimension = tendencies.length;
		for(int i = 0; i < dimension; i++)
		{
			result += "|";
			for(int j = 0; j < dimension; j++)
			{
				result += tendencies[i][j];
				if(j != dimension - 1)
					result += "\t";
			}
			result += "|\n";
		}
		return result;
	}
	
	public void setNumToLabel(int num)
	{
		numToLabel = num;
	}
	
	public int getNumToLabel()
	{
		return numToLabel;
	}
        
        public Worker getWorker()
        {
            return worker;
        }
        
        public double getSim()
        {
            return sim;
        }
        
        public void setSim(double sim)
        {
            this.sim = sim;
        }
        
        public boolean equals(AnalyzedWorker other)
        {
            if(other.getId().equals(this.getId()))
                return true;
            else
                return false;
        }
}
