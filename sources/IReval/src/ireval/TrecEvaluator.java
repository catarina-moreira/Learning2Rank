package ireval;

import ireval.RetrievalEvaluator.Document;
import ireval.RetrievalEvaluator.Judgment;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

@SuppressWarnings("unused")
public class TrecEvaluator {
   
	private TreeMap<Integer, Float> dataMap; 
	private TreeMap<Integer, Float> dataP1;
	private TreeMap<Integer, Float> dataP2;
	private TreeMap<Integer, Float> dataP3;
	private TreeMap<Integer, Float> dataP4;
	private TreeMap<Integer, Float> dataP5;
	private TreeMap<Integer, Float> dataP10;
	private TreeMap<Integer, Float> dataP15;
	private TreeMap<Integer, Float> dataP20;
	
	private TreeMap<Integer, Float> dataNDCG;
	private TreeMap<Integer, Float> dataNDCG1;
	private TreeMap<Integer, Float> dataNDCG2;
	private TreeMap<Integer, Float> dataNDCG3;
	private TreeMap<Integer, Float> dataNDCG4;
	private TreeMap<Integer, Float> dataNDCG5;
	private TreeMap<Integer, Float> dataNDCG10;
	private TreeMap<Integer, Float> dataNDCG15;
	private TreeMap<Integer, Float> dataNDCG20;
	
	private String metric;
	
	public TrecEvaluator ( String metric ){
		
		this.metric = metric;
		
		this.dataMap = new TreeMap<Integer, Float>();
		
		this.dataP1 = new TreeMap<Integer, Float>();
		this.dataP2 = new TreeMap<Integer, Float>();
		this.dataP3 = new TreeMap<Integer, Float>();
		this.dataP4 = new TreeMap<Integer, Float>();
		this.dataP5 = new TreeMap<Integer, Float>();
		this.dataP10 = new TreeMap<Integer, Float>();
		this.dataP15 = new TreeMap<Integer, Float>();
		this.dataP20 = new TreeMap<Integer, Float>();
		
		this.dataNDCG = new TreeMap<Integer, Float>();
		this.dataNDCG1 = new TreeMap<Integer, Float>();
		this.dataNDCG2 = new TreeMap<Integer, Float>();
		this.dataNDCG3 = new TreeMap<Integer, Float>();
		this.dataNDCG4 = new TreeMap<Integer, Float>();
		this.dataNDCG5 = new TreeMap<Integer, Float>();
		this.dataNDCG10 = new TreeMap<Integer, Float>();
		this.dataNDCG15 = new TreeMap<Integer, Float>();
		this.dataNDCG20 = new TreeMap<Integer, Float>();
	}
	
	public static void main(String[ ] args) throws NumberFormatException, IOException{
		
		
		CreateTrecEval eval = new CreateTrecEval( args[0], args[1], Integer.parseInt( args[2] ) );
		
		TrecEvaluator trec = new TrecEvaluator( args[1] );

		for(int i = 1; i <= eval.getNUM_FOLDS( ); i++)
		{
			eval.createTrecEvalFilesSIGR( i );
			String relevanceJudgments = args[0]+"/Fold"+i+"/qRel";
			String rankings = args[0]+"/Fold"+i+"/qResults";
			String results = args[0]+"/Fold"+i +"/";
					
			trec.runTrecEval(relevanceJudgments, rankings , results);	
		}
		trec.readResults( args[0], args[1], eval.getNUM_FOLDS( ) ); 
		
		if( trec.metric.equals( "MAP" ))
			System.out.println("MAP = " + trec.parseMAP( ));
		
		if( trec.metric.equals( "P@1" ))
			System.out.println( "P@1 = " + trec.parseP1( ));
	}
	
	
	
	public TreeMap<Integer, Float> getdataMap( )
	{
		return this.dataMap;
	}
	
	public void readResults(String path, String metric, int NUM_FOLDS) throws NumberFormatException, IOException{
		
		for(int fold = 1; fold <= NUM_FOLDS; fold++){
			String resultFile = path + "/Fold"+fold+"/";
			BufferedReader all = new BufferedReader(new FileReader(resultFile+"results"));		
			
			String temp = "";
			while((temp = all.readLine( )) != null){
				
				String[] tokens = temp.split(" +");
				
				if(tokens[1].contains("all") && tokens[0].contains("map")) {
		
					this.dataMap.put( fold, Float.parseFloat(tokens[2]) );

				}
				
				if(tokens[1].contains("all") && tokens[0].contains("P1") && tokens[0].endsWith("1") ) 
					dataP1.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("P2") && tokens[0].endsWith("2") ) 
					dataP5.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("P3") && tokens[0].endsWith("3")) 
					dataP5.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("P4") && tokens[0].endsWith("4")) 
					dataP5.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("P5") && tokens[0].endsWith("5")) 
					dataP5.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("P10") && tokens[0].endsWith("10") ) 
					dataP10.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("P15") && tokens[0].endsWith("15") ) 
					dataP15.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("P20") && tokens[0].endsWith("20") ) 
					dataP20.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("ndcg") && tokens[0].endsWith("dcg") ) 
					dataNDCG.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("ndcg1") && !tokens[0].contains("dcg1") ) 
					dataNDCG1.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("ndcg2") && !tokens[0].contains("dcg2") ) 
					dataNDCG2.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("ndcg3") && !tokens[0].contains("dcg3") ) 
					dataNDCG3.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("ndcg4") && !tokens[0].contains("dcg4") ) 
					dataNDCG4.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("ndcg5") && !tokens[0].contains("dcg5") ) 
					dataNDCG5.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("ndcg10") && tokens[0].endsWith("10") ) 
					dataNDCG10.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("ndcg15") && tokens[0].endsWith("15") ) 
					dataNDCG15.put(fold, Float.parseFloat(tokens[2]));
				
				if(tokens[1].contains("all") && tokens[0].contains("ndcg20") && tokens[0].endsWith("20") ) 
					dataNDCG20.put(fold, Float.parseFloat(tokens[2]));
			}
			all.close();
		}
	}
	
	public float parseMAP( )
	{
		float result = 0;
		
		for( int i: this.dataMap.keySet( ) )
		{
			System.out.println(dataMap.get( i ));
			result += dataMap.get( i );
		}
		
		System.out.println(dataMap.size( ));
		
		return result/dataMap.size( );
	}
	
	public float parseP1( )
	{
		float result = 0;
		
		for( int i: this.dataP1.keySet( ) )
			result += dataP1.get( i );
		
		return result/dataP1.size( );
	}
	
	/**
     * Loads a TREC judgments file.
     *
     * @param filename The filename of the judgments file to load.
     * @return Maps from query numbers to lists of judgments for each query.
     */
    
    public TreeMap< String, ArrayList<Judgment> > loadJudgments( String filename ) throws IOException, FileNotFoundException {
        // open file
        BufferedReader in = new BufferedReader(new FileReader( filename ));
        String line = null;
        TreeMap< String, ArrayList<Judgment> > judgments = new TreeMap< String, ArrayList<Judgment> >();
        String recentQuery = null;
        ArrayList<Judgment> recentJudgments = null;
        
        while( (line = in.readLine()) != null ) {
            String[] fields = line.split( "\\s" );
            
            String number = fields[0];
			String unused = fields[1];
            String docno = fields[2];
            String judgment = fields[3];
            
            Judgment j = new Judgment( docno, Integer.valueOf( judgment ) );
            
            if( recentQuery == null || !recentQuery.equals( number ) ) {
                if( !judgments.containsKey( number ) ) {
                    judgments.put( number, new ArrayList<Judgment>() );
                }
                
                recentJudgments = judgments.get( number );
                recentQuery = number;
            }
            
            recentJudgments.add( j );
        }

        in.close();
        return judgments;
    }
    
    /**
     * Reads in a TREC ranking file.
     *
     * @param filename The filename of the ranking file.
     * @return A map from query numbers to document ranking lists.
     */

    public TreeMap< String, ArrayList<Document> > loadRanking( String filename ) throws IOException, FileNotFoundException {
        // open file
        BufferedReader in = new BufferedReader(new FileReader( filename ));
        String line = null;
        TreeMap< String, ArrayList<Document> > ranking = new TreeMap< String, ArrayList<Document> >();
        ArrayList<Document> recentRanking = null;
        String recentQuery = null;
        
        while( (line = in.readLine()) != null ) {
            //String[] fields = line.split( "\\s" );
            String[] fields = line.split( "\\s" );
            
            // 1 Q0 WSJ880711-0086 39 -3.05948 Exp
                    
            String number = fields[0];
            String unused = fields[1];
            String docno = fields[2];
            String rank = fields[4];
            String score = fields[3];
            String runtag = fields[5];
            
            Document document = new Document( docno, Integer.valueOf( rank ), Double.valueOf( score ) );
            
            if( recentQuery == null || !recentQuery.equals( number ) ) {
                if( !ranking.containsKey( number ) ) {
                    ranking.put( number, new ArrayList<Document>() );
                }
                
                recentQuery = number;
                recentRanking = ranking.get( number );       
            }
            
            recentRanking.add( document );
        }
        
        in.close();
        return ranking;
    }

    /**
     * Creates a SetRetrievalEvaluator from data from loadRanking and loadJudgments.
     */
    
    public SetRetrievalEvaluator create( TreeMap< String, ArrayList<Document> > allRankings, TreeMap< String, ArrayList<Judgment> > allJudgments ) {
        TreeMap< String, RetrievalEvaluator > evaluators = new TreeMap<String, RetrievalEvaluator>();
        
        for( String query : allRankings.keySet() ) {
            ArrayList<Judgment> judgments = allJudgments.get( query );
            ArrayList<Document> ranking = allRankings.get( query );
            
            if( judgments == null || ranking == null ) {
                continue;
            }
            
            RetrievalEvaluator evaluator = new RetrievalEvaluator( query, ranking, judgments );
            evaluators.put( query, evaluator );
        }
        
        return new SetRetrievalEvaluator( evaluators.values() );
    }
    
    /**
     * When run as a standalone application, this returns output 
     * very similar to that of trec_eval.  The first argument is 
     * the ranking file, and the second argument is the judgments
     * file, both in standard TREC format.
     * @throws IOException 
     */
    public void singleEvaluation( SetRetrievalEvaluator setEvaluator, String pathToWrite ) throws IOException {
    	 String formatString = "%2$-16s%1$3s ";
         
    	 ByteArrayOutputStream baos = new ByteArrayOutputStream();
    	 PrintStream ps = new PrintStream(baos);
    	 
         // print trec_eval relational-style output
         for( RetrievalEvaluator evaluator : setEvaluator.getEvaluators() ) {
             String query = evaluator.queryName();
             
             // counts
             ps.format( formatString + "%3$d\n",        query, "num_ret",     evaluator.retrievedDocuments().size() );
             ps.format( formatString + "%3$d\n",        query, "num_rel",     evaluator.relevantDocuments().size() );
             ps.format( formatString + "%3$d\n",        query, "num_rel_ret", evaluator.relevantRetrievedDocuments().size() );

             // aggregate measures
             ps.format( formatString + "%3$6.4f\n",     query, "map",         evaluator.averagePrecision() );
             ps.format( formatString + "%3$6.4f\n",     query, "ndcg",        evaluator.normalizedDiscountedCumulativeGain() );
             ps.format( formatString + "%3$6.4f\n",     query, "ndcg1",        evaluator.normalizedDiscountedCumulativeGain( 1 ) );
             ps.format( formatString + "%3$6.4f\n",     query, "ndcg2",        evaluator.normalizedDiscountedCumulativeGain( 2 ) );
             ps.format( formatString + "%3$6.4f\n",     query, "ndcg3",        evaluator.normalizedDiscountedCumulativeGain( 3 ) );
             ps.format( formatString + "%3$6.4f\n",     query, "ndcg4",        evaluator.normalizedDiscountedCumulativeGain( 4 ) );
             ps.format( formatString + "%3$6.4f\n",     query, "ndcg5",       evaluator.normalizedDiscountedCumulativeGain( 5 ) );
             ps.format( formatString + "%3$6.4f\n",     query, "ndcg10",      evaluator.normalizedDiscountedCumulativeGain( 10 ) );
             ps.format( formatString + "%3$6.4f\n",     query, "ndcg15",      evaluator.normalizedDiscountedCumulativeGain( 15 ) );
             ps.format( formatString + "%3$6.4f\n",     query, "ndcg20",      evaluator.normalizedDiscountedCumulativeGain( 20 ) );

             ps.format( formatString + "%3$6.4f\n",     query, "R-prec",      evaluator.rPrecision() );
             ps.format( formatString + "%3$6.4f\n",     query, "bpref",       evaluator.binaryPreference() );
             ps.format( formatString + "%3$6.4f\n",     query, "recip_rank",  evaluator.reciprocalRank() );
             
             // precision at fixed points
             int[] fixedPoints = { 1,2,3,4,5, 10, 15, 20, 30, 100, 200, 500, 1000 };
             
             for( int i=0; i<fixedPoints.length; i++ ) {
                 int point = fixedPoints[i];
                 ps.format( formatString + "%3$6.4f\n", query, "P" + point,   evaluator.precision( fixedPoints[i] ) );
             }
         }
         
         // print summary data
         ps.format( formatString + "%3$d\n",      "all", "num_ret",     setEvaluator.numberRetrieved() );
         ps.format( formatString + "%3$d\n",      "all", "num_rel",     setEvaluator.numberRelevant() );
         ps.format( formatString + "%3$d\n",      "all", "num_rel_ret", setEvaluator.numberRelevantRetrieved() );
         
         ps.format( formatString + "%3$6.4f\n",   "all", "map",         setEvaluator.meanAveragePrecision() );
         ps.format( formatString + "%3$6.4f\n",   "all", "ndcg",        setEvaluator.meanNormalizedDiscountedCumulativeGain() );
         ps.format( formatString + "%3$6.4f\n",   "all", "ndcg5",       setEvaluator.meanNormalizedDiscountedCumulativeGain1( ) );
         ps.format( formatString + "%3$6.4f\n",   "all", "ndcg5",       setEvaluator.meanNormalizedDiscountedCumulativeGain2( ) );
         ps.format( formatString + "%3$6.4f\n",   "all", "ndcg5",       setEvaluator.meanNormalizedDiscountedCumulativeGain3( ) );
         ps.format( formatString + "%3$6.4f\n",   "all", "ndcg5",       setEvaluator.meanNormalizedDiscountedCumulativeGain4( ) );
         ps.format( formatString + "%3$6.4f\n",   "all", "ndcg5",       setEvaluator.meanNormalizedDiscountedCumulativeGain5( ) );
         ps.format( formatString + "%3$6.4f\n",   "all", "ndcg10",      setEvaluator.meanNormalizedDiscountedCumulativeGain10( ) );
         ps.format( formatString + "%3$6.4f\n",   "all", "ndcg15",      setEvaluator.meanNormalizedDiscountedCumulativeGain15( ) );
         ps.format( formatString + "%3$6.4f\n",   "all", "ndcg20",      setEvaluator.meanNormalizedDiscountedCumulativeGain20( ) );
         
         ps.format( formatString + "%3$6.4f\n",   "all", "R-prec",      setEvaluator.meanRPrecision() );
         ps.format( formatString + "%3$6.4f\n",   "all", "bpref",       setEvaluator.meanBinaryPreference() );
         ps.format( formatString + "%3$6.4f\n",   "all", "recip_rank",  setEvaluator.meanReciprocalRank() );
         
         // precision at fixed points
         int[] fixedPoints = { 1,2,3,4,5, 10, 15, 20, 30, 100, 200, 500, 1000 };

         for( int i=0; i<fixedPoints.length; i++ ) {
             int point = fixedPoints[i];
             ps.format( formatString + "%3$6.4f\n", "all", "P" + point,   setEvaluator.meanPrecision( fixedPoints[i] ) );
         }
         BufferedWriter bw = new BufferedWriter(new FileWriter(pathToWrite+"results"));
         bw.write(baos.toString("ISO-8859-1").replace(",", "."));
         bw.close();
    }

    /**
     * Compares two ranked lists with statistical tests on most major metrics.
     */

    
    public void comparisonEvaluation( SetRetrievalEvaluator baseline, SetRetrievalEvaluator treatment ) {
        String[] metrics = { "averagePrecision", "P5", "P10", "P15", "P20" };
        String formatString = "%1$-20s%2$-12s%3$6.4f\n";
        String integerFormatString = "%1$-20s%2$-12s%3$d\n";
        
        for( String metric : metrics ) {
            Map<String, Double> baselineMetric = baseline.evaluateAll( metric );
            Map<String, Double> treatmentMetric = treatment.evaluateAll( metric );
            
            SetRetrievalComparator comparator = new SetRetrievalComparator( baselineMetric, treatmentMetric );

            System.out.format( formatString, metric, "baseline", comparator.meanBaselineMetric() );
            System.out.format( formatString, metric, "treatment", comparator.meanTreatmentMetric() );
            
            System.out.format( integerFormatString, metric, "basebetter", comparator.countBaselineBetter() );
            System.out.format( integerFormatString, metric, "treatbetter", comparator.countTreatmentBetter() );
            System.out.format( integerFormatString, metric, "equal", comparator.countEqual() );
            
            System.out.format( formatString, metric, "ttest", comparator.pairedTTest() );
            System.out.format( formatString, metric, "randomized", comparator.randomizedTest() );
            System.out.format( formatString, metric, "signtest", comparator.signTest() );
        }
    }
    
    public void usage( ) {
        System.err.println( "ireval: " );
        System.err.println( "   There are two ways to use this program.  First, you can evaluate a single ranking: " );
        System.err.println( "      java -jar ireval.jar TREC-Ranking-File TREC-Judgments-File" );
        System.err.println( "   or, you can use it to compare two rankings with statistical tests: " );
        System.err.println( "      java -jar ireval.jar TREC-Baseline-Ranking-File TREC-Improved-Ranking-File TREC-Judgments-File" );
        System.exit(-1);
    }
    
    public void runTrecEval( String fileJudgments, String fileRankings, String pathToWrite)  {
        try {
                TreeMap< String, ArrayList<Document> > ranking = loadRanking( fileRankings );
                TreeMap< String, ArrayList<Judgment> > judgments = loadJudgments( fileJudgments );

                SetRetrievalEvaluator setEvaluator = create( ranking, judgments );
                singleEvaluation( setEvaluator,pathToWrite );
            
        } catch( IOException e ) {
            e.printStackTrace();
            usage();
        }
    }
}
