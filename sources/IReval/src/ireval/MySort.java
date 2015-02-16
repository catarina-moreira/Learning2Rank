package ireval;

import java.util.HashMap;
import java.util.TreeMap;

public class MySort
{
	/**
	 * 
	 * @param map
	 * @return
	 */
	public TreeMap<String, Double> sort(HashMap<String, Double> map){

		ValueComparator bvc =  new ValueComparator(map);
		TreeMap<String,Double> sorted_map = new TreeMap<String, Double>(bvc);

		sorted_map.putAll(map);

		return sorted_map;
	}
	
}
