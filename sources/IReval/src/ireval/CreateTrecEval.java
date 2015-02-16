package ireval;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.TreeMap;



public class CreateTrecEval{
    
    
    private String filePath;
    private String metric;
    
    private int NUM_FOLDS;
    
    public CreateTrecEval(String filePath, String metric, int numFolds){
        this.filePath = filePath;
        this.metric = metric;
        this.NUM_FOLDS = numFolds;
    }
    
    @SuppressWarnings( "resource" )
	public void createTrecEvalFilesSIGR(int fold) {
        
	    HashMap<String, Double> map = new HashMap<String, Double>();
	    TreeMap<String, Double> tree = new TreeMap<String, Double>();
        
	    Double val1 = 0.0;
	    
		try {

			BufferedReader t = new BufferedReader(new FileReader(getFilePath()+"/Fold"+fold+"/test"));
			BufferedReader p = new BufferedReader(new FileReader(getFilePath()+"/Fold"+fold+"/prediction"));
			
			File f = new File(getFilePath() +"/Fold"+fold+"/qRel");
			f.delete();
			
			f = new File(getFilePath() +"/Fold"+ fold+"/qResults");
			f.delete();
			
			BufferedWriter qRel = new BufferedWriter(new FileWriter(getFilePath()+"/Fold"+fold+"/qRel"));
			BufferedWriter qResults = new BufferedWriter(new FileWriter(getFilePath()+"/Fold"+fold+"/qResults"));
			
			String lineTestFile = new String(), linePredictionFile = new String(), l = new String();
			
			int j =0;
			
			while(((lineTestFile = t.readLine()) != null) && ((linePredictionFile = p.readLine()) != null)){
				
				String[] token = lineTestFile.split("\\s");
				
				if(!l.equals(lineTestFile.split("\\s")[1])) {
                    
					MySort myS = new MySort();
					
					j = 0; l = lineTestFile.split("\\s")[1]; tree = myS.sort(map); 
					
					
					for(String s: tree.keySet() ) {
						double result = 0.0;
						try{
                            result = tree.get(s);
                            
						}catch( java.lang.NullPointerException e) { result = map.get(s); }
						
						qRel.write(s.split("\t")[1] + "\t" + 0 + "\t" + s.split("\t")[0] + "\t" + s.split("\t")[2] + "\n"); 
						qResults.write(s.split("\t")[1] + "\t" + "Q0" + "\t" + s.split("\t")[0] + "\t" + result  + "\t" + ++j + "\t" + "whatever" + "\n");
					}
					map = new HashMap<String, Double>();
				}
				
				val1 = Double.parseDouble(linePredictionFile);
				
			
				map.put(token[token.length-2].substring(1)+"_"+token[token.length-1]+"\t"+token[1].replace("qid:", "")+"\t"+token[0], val1);
			}
			
			qRel.close();
			qResults.close();
			getLastExtrySigr(getFilePath(), map, fold);
			
		}catch (IOException e) { System.out.println("[createTrecEvalFiles] Could Not Find File!"); e.printStackTrace(); }
        
    }
    
    
    public void getLastExtrySigr(String path,  HashMap<String, Double> map, int fold) throws IOException{
  		TreeMap<String, Double> tree = new TreeMap<String, Double>();
  		MySort u = new MySort();
		tree = u.sort(map); 
		
		int j = 0;
		BufferedWriter qRel = new BufferedWriter(new FileWriter(path +"/Fold"+fold+"/qRel", true));
		BufferedWriter qResults = new BufferedWriter(new FileWriter(path +"/Fold"+fold+"/qResults", true));
        
		for(String s: tree.keySet()){
            
			double result = 0.0;
			try{
                result = tree.get( s);
                
			}catch( java.lang.NullPointerException e) { result = map.get(s); }
			
			qRel.write(s.split("\t")[1] + "\t" + 0 + "\t" + s.split("\t")[0] + "\t" + s.split("\t")[2] + "\n"); 
			qResults.write(s.split("\t")[1] + "\t" + "Q0" + "\t" + s.split("\t")[0] + "\t" + result  + "\t" + ++j + "\t" + "whatever" + "\n");
		}
		qRel.close();
		qResults.close();
	}
    
    @SuppressWarnings( "resource" )
	public double parseResults() throws IOException{
    	
    	double map = 0;
    	
    	for(int i = 1; i <= NUM_FOLDS; i++){
    		
    		BufferedReader reader = new BufferedReader(new FileReader(getFilePath() +"/Fold"+i +"/results"));
    		String line = new String();
    		
    		while( (line = reader.readLine()) != null ){
    			
    			if( line.contains(metric.replace( "@", "" )) && line.contains("all") )
    				map += Double.parseDouble(line.split("\\s+")[2]);
    		}
    	}
    	return map/NUM_FOLDS;
    	
    }
    
    
    public String getFilePath(){
        return this.filePath;
    }
    
    public void setFilePath(String newPath){
        this.filePath = newPath;
    }
    
    public int getNUM_FOLDS( )
    {
    	return this.NUM_FOLDS;
    }

}






