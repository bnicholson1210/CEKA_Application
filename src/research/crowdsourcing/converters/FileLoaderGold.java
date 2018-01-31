package crowdsourcing.converters;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;

import weka.core.FastVector;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;
import ceka.utils.Misc;

public class FileLoaderGold 
{
	public static Dataset loadFileGold(String goldPath) throws Exception {
		String relationName = Misc.exstractFileName(goldPath, false);
		FastVector attInfo = new FastVector();
		int capacity = 0;
		Dataset dataset = new Dataset(relationName, attInfo, capacity);
		
		ArrayList<Integer> categories = new ArrayList<Integer>();
		
		// read gold file
		FileReader reader = new FileReader(goldPath);
		BufferedReader readerBuffer = new BufferedReader(reader);
		String line = null;
		
		while((line = readerBuffer.readLine()) != null) {
			String [] subStrs = line.split("[ \t]");
			String exampleId = subStrs[0];
			Example example = null;
			if ((example = dataset.getExampleById(exampleId)) != null) {
				Label trueLabel = new Label(null, subStrs[1], exampleId, Worker.WORKERID_GOLD);
				example.setTrueLabel(trueLabel);
			}
			else {
				example = new Example(1, exampleId);
				Label trueLabel = new Label(null, subStrs[1], exampleId, Worker.WORKERID_GOLD);
				example.setTrueLabel(trueLabel);
				dataset.addExample(example);
			}
			Misc.addElementIfNotExistedEquals(categories, Integer.parseInt(subStrs[1]));
		}
		readerBuffer.close();
		reader.close();
		
		// check and categories
		Collections.sort(categories);
		boolean correct = true;
		for (int k = 0; k < categories.size(); k++) {
			if (categories.get(k).intValue() != k) {
				correct = false;
				break;
			}
			Category category = new Category(null, categories.get(k).toString());
			dataset.addCategory(category);
		}
		if (!correct)
			throw new Exception("Invalid cateories, categories must be consecutive integers staring from 0");
		
		return dataset;
	}
}
