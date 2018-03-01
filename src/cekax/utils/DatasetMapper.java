
package cekax.utils;

import ceka.core.Dataset;
import ceka.core.Example;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DatasetMapper<T> {
    private Map<String, T> exampleMap;
    
    public DatasetMapper(Dataset dataset, List<T> list) throws IllegalStateException{
        exampleMap = new HashMap<>();
        if(list.size() != dataset.getExampleSize()){
            throw new IllegalStateException("Could not map dataset and list together: " +
                    "number of examples in dataset different from number of elements in list.");
        }
        for(int i = 0; i < dataset.getExampleSize(); i++){
            exampleMap.put(dataset.getExampleByIndex(i).getId(), list.get(i));
        }
    }
    
    public T getAssociatedObjectOfExample(Example e) throws Exception{
        T object = exampleMap.get(e.getId());
        if(object == null){
            throw new Exception("No object found associated with Example " + e.getId());
        }
        return object;
    }
}
