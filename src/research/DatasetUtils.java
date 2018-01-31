/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package research;

import ceka.core.Dataset;
import ceka.core.Example;

public class DatasetUtils {
 
    	public static Dataset combineDatasets(Dataset dataset1, Dataset dataset2)
	{
            Dataset result = dataset1.generateEmpty();
            for(int i = 0; i < dataset1.getCategorySize(); i++)
            {
                    result.addCategory(dataset1.getCategory(i));
            }
            for(int i = 0; i < dataset1.getExampleSize(); i++)
            {
                    result.addExample(dataset1.getExampleByIndex(i));
            }
            int offset = result.getExampleSize();
            for(int i = 0; i < dataset2.getExampleSize(); i++)
            {
                Example e = dataset2.getExampleByIndex(i);
                e.setId("" + (Integer.parseInt(e.getId()) + offset));
                result.addExample(e);
            }
            return result;
	}
        
        public static Dataset makeCopy(Dataset dataset)
	{
            Dataset result = dataset.generateEmpty();
            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
                result.addExample(dataset.getExampleByIndex(i));
            }
            for(int i = 0; i < dataset.getWorkerSize(); i++)
            {
                result.addWorker(dataset.getWorkerByIndex(i));
            }
            for(int i = 0; i < dataset.getCategorySize(); i++)
            {
                result.addCategory(dataset.getCategory(i));
            }
            return result;
	}
}
