package ireval;

import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class ValueComparator implements Comparator<Object> {

	Map<String, Double> base;
	
	public ValueComparator(Map<String, Double> base) {
		this.base = base;
	}
	
	
	public int compare(Object a, Object b) {

		if((Double)base.get(a) < (Double)base.get(b)) {
			return 1;
		} else if((Double)base.get(a) == (Double)base.get(b)) {
			return 0;
		} else {
			return -1;
		}
	}

	public TreeMap<String, Double> sort(HashMap<String, Double> oldMap){

		TreeMap<String,Double> sorted_map = new TreeMap<String, Double>(this);
		TreeMap<String,Double> sortedMap = new TreeMap<String, Double>(this);
		sorted_map.putAll(this.getBase());
		sortedMap.putAll(this.getBase());

		for(String s: sorted_map.keySet()){
			
			if(sorted_map.get(s)==null)
				sortedMap.put(s, oldMap.get(s)); 
		}

		return sortedMap;
	}

	/**
	 * @return the base
	 */
	public Map<String, Double> getBase() {
		return base;
	}

	/**
	 * @param base the base to set
	 */
	public void setBase(Map<String, Double> base) {
		this.base = base;
	}
}