package research.crowdsourcing;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Scanner;

public class TextParser 
{
	public static void createLatexCode(File f) throws Exception
	{
		DecimalFormat df = new DecimalFormat("#.0000");
		double datasetColWidth = 2.2;
		ArrayList<String> GMMTableCode = new ArrayList();
		ArrayList<String> AccTableCode = new ArrayList();
		String AccTableCodePrefix = "\\begin{table}\n\\begin{center}\n\\begin{tabular}{" +
				"|m{" + datasetColWidth + "cm}|m{1.3cm}|m{1.3cm}|m{1.3cm}|}\n\\hline\n" +
				"\\textbf{Data Set} & \\textbf{True Acc.} & \\textbf{Init. Error} & \\textbf{Final Error}\\\\\n\\hline\n";
		String GMMTableCodePrefix = "\\begin{table}\n\\begin{center}\n\\begin{tabular}{" +
				"|m{" + datasetColWidth + "cm}|m{1.2cm}|m{1.2cm}|m{1.2cm}|m{1.2cm}|m{1.2cm}|" +
				"m{1.2cm}|m{1.2cm}|m{1.2cm}|m{1.2cm}|}\n\\hline\n" +
				"\\textbf{Data Set} & \\textbf{True Acc.} & \\textbf{$GMM_{1}$ Error} & \\textbf{$GMM_{2}$ Error}" +
				"& \\textbf{$GMM_{3}$ Error} & \\textbf{$GMM_{4}$ Error} & \\textbf{$GMM_{5}$ Error} & \\textbf{$GMM_{6}$ Error}"
				+ "& \\textbf{$GMM_{7}$ Error} & \\textbf{$GMM_{8}$ Error}\\\\\n\\hline\n";
		String GMMTable = GMMTableCodePrefix;
		String AccTable = AccTableCodePrefix;
		Scanner s = new Scanner(f);
		ArrayList<ArrayList<Double>> GMMAcc = new ArrayList();
		while(s.hasNextLine())
		{
			ArrayList<Double> GMM = new ArrayList();
			//First line is 
			//Analyzing C:\\ ...
			String datasetPathLine = s.nextLine();
			String datasetPath = datasetPathLine.split(" ")[1];
			String datasetName = getDatasetName(datasetPath);
			//Next 3 lines are meaningless
			s.nextLine();
			s.nextLine();
			s.nextLine();
			//Next line is accuracy of labels generated from initial tendency interrelation model
			double timInitAcc = Double.parseDouble(s.nextLine().split(" ")[0]);
			//Next line is accuracy of labels from real workers
			double trueAcc = Double.parseDouble(s.nextLine().split(" ")[0]);
			GMMTable += datasetName + " & " + df.format(trueAcc);
			AccTable += datasetName + " & " + df.format(trueAcc);
			//Next 3 lines are meaningless
			s.nextLine();
			s.nextLine();
			s.nextLine();
			//Next line is accuracy of labels generated from final tendency interrelation model
			double timFinalAcc = Double.parseDouble(s.nextLine().split(" ")[0]);
			double initError = Math.abs(timInitAcc - trueAcc) / trueAcc;
			double finalError = Math.abs(timFinalAcc - trueAcc) / trueAcc;
			AccTable += " & " + df.format(initError) + " & " + df.format(finalError) + "\\\\\n\\hline\n";
			//Next line is true accuracy, already acquired
			s.nextLine();
			//Array to hold the 8 values representing the accuracy of the GMM models with 
			//varying # of components
			//Loop through each of the 8 segments displaying GMM results. Most lines are meaningless.
			for(int i = 0; i < 8; i++)
			{
				s.nextLine();
				s.nextLine();
				s.nextLine();
				s.nextLine();
				double gmm = Double.parseDouble(s.nextLine().split(" ")[0]);
				gmm = Math.abs(gmm - trueAcc) / trueAcc;
				GMM.add(gmm);
				s.nextLine();
				GMMTable += " & " + df.format(gmm);
			}
			GMMTable += "\\\\\n\\hline\n";
			GMMAcc.add(GMM);
		}
		GMMTable += "\\textbf{Total} &";
		for(int i = 0; i < GMMAcc.get(0).size(); i++)
		{
			double total = 0;
			for(int j = 0; j < GMMAcc.size(); j++)
			{
				total += GMMAcc.get(j).get(i);
				System.out.println(GMMAcc.get(j).get(i));
			}
			System.out.println(total);
			GMMTable += " & " + df.format(total);
		}
		GMMTable += "\\\\\n\\hline\n";
		GMMTable += "\\end{tabular}\n\\caption{Error of GMMs with Respect to True Label Accuracy}\n\\end{center}\n\\end{table}\n";
		AccTable += "\\end{tabular}\n\\caption{Error of Tendency Interrelation Model with Respect to True Label Accuracy}\n\\end{center}\n\\end{table}\n";
		File file = new File("latex.txt");
		BufferedWriter bw = new BufferedWriter(new FileWriter(file));
		bw.write(AccTable);
		bw.write(GMMTable);
		bw.close();
	}
	
	private static String getDatasetName(String datasetPath)
	{
		if(datasetPath.equals("C:\\CekaApp\\Ceka\\data\\real-world\\income94crowd\\arff\\income94HighAccGold.arff"))
			return "Income 94";
		if(datasetPath.equals("C:\\CekaApp\\Ceka\\data\\real-world\\leaves\\arff\\alder.arff"))
			return "Alder";
		if(datasetPath.equals("C:\\CekaApp\\Ceka\\data\\real-world\\leaves\\arff\\eucalyptus.arff"))
			return "Eucalyptus";
		if(datasetPath.equals("C:\\CekaApp\\Ceka\\data\\real-world\\leaves\\arff\\maple.arff"))
			return "Maple";
		if(datasetPath.equals("C:\\CekaApp\\Ceka\\data\\real-world\\leaves\\arff\\oak.arff"))
			return "Oak";
		if(datasetPath.equals("C:\\CekaApp\\Ceka\\data\\real-world\\leaves\\arff\\poplar.arff"))
			return "Poplar";
		if(datasetPath.equals("C:\\CekaApp\\Ceka\\data\\real-world\\leaves\\arff\\tilia.arff"))
			return "Tilia";
		if(datasetPath.equals("C:\\CekaApp\\Ceka\\data\\real-world\\eastpolitics\\eastpolitics2000AMT\\eastpolitics.gold.txt"))
			return "EP 2000";
		if(datasetPath.equals("C:\\CekaApp\\Ceka\\data\\real-world\\eastpolitics\\simulated1000\\eastpoliticsSub1000.gold.txt"))
			return "EP 1000";
		return "???";
		
	}
}
