package uk.ac.ed.easyccg.syntax;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import uk.ac.ed.easyccg.main.EasyCCG.InputFormat;
import uk.ac.ed.easyccg.syntax.Combinator.RuleProduction;
import uk.ac.ed.easyccg.syntax.Combinator.RuleType;
import uk.ac.ed.easyccg.syntax.InputReader.InputToParser;
import uk.ac.ed.easyccg.syntax.SyntaxTreeNode.SyntaxTreeNodeBinary;
import uk.ac.ed.easyccg.syntax.SyntaxTreeNode.SyntaxTreeNodeFactory;
import uk.ac.ed.easyccg.syntax.SyntaxTreeNode.SyntaxTreeNodeLeaf;
import uk.ac.ed.easyccg.syntax.SyntaxTreeNode.SyntaxTreeNodeUnary;
import uk.ac.ed.easyccg.syntax.SyntaxTreeNode.SyntaxTreeNodeVisitor;

import com.google.common.base.Stopwatch;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;

public class ParserAStar implements Parser
{

  public ParserAStar(
      Tagger tagger, int maxSentenceLength, int nbest, double nbestBeam,
      InputFormat inputFormat, List<String> validRootCategories,
      Collection<Combinator> standardCombinators, File unaryRulesFile,
      File extraCombinatorsFile, File seenRulesFile) 
  throws IOException {
    this.tagger = tagger;
    this.maxLength = maxSentenceLength;
    this.nbest = nbest;
    this.nodeFactory = new SyntaxTreeNodeFactory(maxSentenceLength, 0);
    this.reader = InputReader.make(inputFormat, nodeFactory);
    this.unaryRules = loadUnaryRules(unaryRulesFile);
    this.seenRules = new SeenRules(seenRulesFile);
    this.nbestBeam = Math.log(nbestBeam);
    
    List<Combinator> combinators = new ArrayList<>(standardCombinators);
    
    if (extraCombinatorsFile.exists()) {
      combinators.addAll(Combinator.loadSpecialCombinators(extraCombinatorsFile));
    }
    this.binaryRules = ImmutableList.copyOf(combinators);    
    
    List<Category> cats = new ArrayList<Category>();
    for (String cat : validRootCategories) {
      cats.add(Category.valueOf(cat));
    }
    possibleRootCategories = ImmutableSet.copyOf(cats);
  }
  
  private final int maxLength;
  
  private final Collection<Combinator> binaryRules;
  private final Multimap<Category, Category> unaryRules;
  private final int nbest;
  private final double nbestBeam;
  
  private final Tagger tagger;
  private final SyntaxTreeNodeFactory nodeFactory;

  private final InputReader reader;
  
  private final SeenRules seenRules;

  private final Collection<Category> possibleRootCategories;
  
  // Included to try to help me suck less at log probabilities...
  private final static double CERTAIN = 0.0;
  private final static double IMPOSSIBLE = Double.NEGATIVE_INFINITY;
  
  private final Multimap<Integer, Long> sentenceLengthToParseTimeInNanos = HashMultimap.create();
  
  @Override
  public Multimap<Integer, Long> getSentenceLengthToParseTimeInNanos()
  {
    return sentenceLengthToParseTimeInNanos;
  }

  private final Stopwatch parsingTimeOnly = Stopwatch.createUnstarted();
  private final Stopwatch taggingTimeOnly = Stopwatch.createUnstarted();

  /**
   * Loads file containing unary rules
   */
  private Multimap<Category, Category> loadUnaryRules(File file) throws IOException {
    Multimap<Category, Category> result = HashMultimap.create();
    for (String line : Util.readFile(file)) {
      // Allow comments.
      if (line.indexOf("#") > -1) line = line.substring(0, line.indexOf("#"));
      line = line.trim();
      if (line.isEmpty()) continue;
     
      String[] fields = line.split("\\s+");
      if (fields.length != 2) throw new Error("Expected 2 categories on line in UnaryRule file: " + line);
      result.put(Category.valueOf(fields[0]), Category.valueOf(fields[1]));
    }
    
    return ImmutableMultimap.copyOf(result);
  }


  /* (non-Javadoc)
   * @see uk.ac.ed.easyccg.syntax.ParserInterface#parse(java.lang.String)
   */
  @Override
  public List<SyntaxTreeNode> parse(String line) {
    InputToParser input = reader.readInput(line);
    List<SyntaxTreeNode> parses = doParsing(input);   
    return parses;
  }
  
  /* (non-Javadoc)
   * @see uk.ac.ed.easyccg.syntax.ParserInterface#parse(java.lang.String)
   */
  @Override
  public List<SyntaxTreeNode> parse(SuperTaggingResults results, String line) {
    InputToParser input = reader.readInput(line);
    List<SyntaxTreeNode> parses = parseSentence(results, input);   
    return parses;
  }

  /* (non-Javadoc)
   * @see uk.ac.ed.easyccg.syntax.ParserInterface#parseTokens(java.util.List)
   */
  @Override
  public List<SyntaxTreeNode> parseTokens(List<String> words) {
    InputToParser input = InputToParser.fromTokens(words);
    List<SyntaxTreeNode> parses = doParsing(input);
    
    return parses;
  }
  
  public static class SuperTaggingResults {
    public AtomicInteger parsedSentences = new AtomicInteger();
    public AtomicInteger totalSentences = new AtomicInteger();
    
    public AtomicInteger rightCats = new AtomicInteger();
    public AtomicInteger totalCats = new AtomicInteger();
    
    public AtomicInteger exactMatch = new AtomicInteger();    
  }
  
  /* (non-Javadoc)
   * @see uk.ac.ed.easyccg.syntax.ParserInterface#parseFile(java.io.File, uk.ac.ed.easyccg.syntax.Parser.SuperTaggingResults)
   */
  @Override
  public Iterator<List<SyntaxTreeNode>> parseFile(File file, final SuperTaggingResults results) throws IOException {

    final Iterator<InputToParser> input = reader.readFile(file).iterator();
    return new Iterator<List<SyntaxTreeNode>>() {

      @Override
      public boolean hasNext()
      {
        return input.hasNext();
      }

      @Override
      public List<SyntaxTreeNode> next()
      {
        return parseSentence(results, input.next());
      }

      @Override
      public void remove()
      {
        throw new UnsupportedOperationException();
      }
    };
    
  }

  /* (non-Javadoc)
   * @see uk.ac.ed.easyccg.syntax.ParserInterface#parseSentence(uk.ac.ed.easyccg.syntax.Parser.SuperTaggingResults, uk.ac.ed.easyccg.syntax.InputReader.InputToParser)
   */
  @Override
  public List<SyntaxTreeNode> parseSentence(SuperTaggingResults results,
      InputToParser input)
  {
    results.totalSentences.incrementAndGet();

    if (input.length() >= maxLength) {
      System.err.println("Skipping sentence of length " + input.length());
      return null;
    }
    
    List<SyntaxTreeNode> parses = doParsing(input);
    
    if (parses != null) {
      results.parsedSentences.incrementAndGet();

      if (input.haveGoldCategories()) {
        List<Category> gold = input.getGoldCategories();
        
        int bestCorrect = -1;

        for (SyntaxTreeNode parse : parses) {
          List<Category> supertags = getSupertags(parse);
          int correct = countCorrect(gold, supertags);
          if (correct > bestCorrect) {
            bestCorrect = correct;
          }
        }
        
        results.rightCats.addAndGet(bestCorrect);
        results.totalCats.addAndGet(gold.size());
        
        if (bestCorrect == gold.size()) {
          results.exactMatch.incrementAndGet();
        }
      }

    } else {
      System.err.println("FAILED TO PARSE! " + input.getWordsAsString());
    }
    return parses;
  }

  /* (non-Javadoc)
   * @see uk.ac.ed.easyccg.syntax.ParserInterface#doParsing(uk.ac.ed.easyccg.syntax.InputReader.InputToParser)
   */
  @Override
  public List<SyntaxTreeNode> doParsing(InputToParser input)
  {
    if (input.length() > maxLength) {
      System.err.println("Skipping sentence of length " + input.length());
      return null;
    }
    
    Stopwatch stopWatch = Stopwatch.createStarted();
    
    List<List<SyntaxTreeNodeLeaf>> supertags;
    
    if (input.isAlreadyTagged()) {
      supertags = input.getInputSupertags();
    } else {
      try{
        taggingTimeOnly.start();
        supertags = tagger.tag(input.getInputWords());
      } finally {
        taggingTimeOnly.stop();
      }
    }
    
    // For tokens that are constrained to a specific supertag, replace the
    // supertag distributions accordingly.
    if (input.getSupertagConstraints() != null) {
      for (int i = 0; i < input.length(); i++) {
        List<SyntaxTreeNodeLeaf> constraint = input.getSupertagConstraints().get(i);
        
        if (constraint != null) {
          supertags.remove(i);
          supertags.add(i, constraint);
        }
      }
    }
    
    try{
      parsingTimeOnly.start();
      List<SyntaxTreeNode> parses = parseAstar(supertags, input.getSpanConstraints());
      return parses;
    } finally {
      parsingTimeOnly.stop();
      stopWatch.stop();    
      sentenceLengthToParseTimeInNanos.put(input.length(), stopWatch.elapsed(TimeUnit.NANOSECONDS));
    }
  }

  
  private int countCorrect(List<Category> gold, List<Category> predicted)
  {
    int rightCats = 0;
    for (int i=0; i<gold.size(); i++) {
      if (predicted.get(i).equals(gold.get(i))) {
        rightCats++;
      }
    }
    return rightCats;
  }


  static class AgendaItem implements Comparable<AgendaItem> {
    private final SyntaxTreeNode parse;
    AgendaItem(SyntaxTreeNode parse, double outsideProbabilityUpperBound, int startOfSpan, int spanLength)
    {
      this.parse = parse;
      this.startOfSpan = startOfSpan;
      this.spanLength = spanLength;
      this.cost = parse.probability + outsideProbabilityUpperBound;
      
    }
    private final int startOfSpan;
    private final int spanLength;
    private final double cost;
    
    /**
     * Comparison function used to order the agenda.
     */
    @Override
    public int compareTo(AgendaItem o)
    {
      int result = Double.compare(o.cost, cost);
      
      if (result != 0 && Math.abs(o.cost - cost) < 0.0000001) {
        // Allow some tolerance on comparisons of Doubles.
        result = 0;
      }
      
      // Commented out below is the tie-breaking heuristic from the original
      // EasyCCG paper. Confusingly, the implementation does the opposite of
      // what the comment and the paper say, it prefers *shorter* dependencies!
      
      /*if (result == 0) {
        // All other things being equal, it works best to prefer parser with longer dependencies (i.e. non-local attachment).
        result = parse.totalDependencyLength - o.parse.totalDependencyLength;
      }*/
      
      // We use this heuristic instead:
      
      if (result == 0) {
          // All other things being equal, we want fewer arguments to be delayed.
          result = parse.argumentsDelayed - o.parse.argumentsDelayed;
      }
      
      if (result == 0) {
          // All other things being equal, we want fewer type raising rules.
          result = parse.typeRaisingRulesUsed - o.parse.typeRaisingRulesUsed;
      }

      return result;
    }
    
    @Override
    public String toString()
    {
        return startOfSpan + " " + spanLength + " " + parse.getCategory() + " " + cost + " " + parse.totalDependencyLength + " " + parse.typeRaisingRulesUsed + " " + parse.argumentsDelayed;
    }
  } 

  /**
   * Takes supertagged input and returns a set of parses.
   * 
   * Returns null if the parse fails.
   */
  private List<SyntaxTreeNode> parseAstar(List<List<SyntaxTreeNodeLeaf>> supertags, SpanConstraints spanConstraints) {

    final int sentenceLength = supertags.size();
    final PriorityQueue<AgendaItem> agenda = new PriorityQueue<ParserAStar.AgendaItem>();
    final ChartCell[][] chart = new ChartCell[sentenceLength][sentenceLength];

    final double[][] outsideProbabilitiesUpperBound = computeOutsideProbabilities(supertags);
    
    
    for (int word = 0; word < sentenceLength; word++) {
      for (SyntaxTreeNode entry : supertags.get(word)) {
        agenda.add(new AgendaItem(entry, outsideProbabilitiesUpperBound[word][word + 1], word, 1));
      }
    }
        
    int parsesFound = 0;
   
    while (chart[0][sentenceLength - 1] == null || 
           (parsesFound < nbest 
           && agenda.peek() != null && agenda.peek().cost > chart[0][sentenceLength - 1].bestValue + nbestBeam
           )) {
      // Add items from the agenda, until we have enough parses.
      
      final AgendaItem agendaItem = agenda.poll();
      if (agendaItem == null) {
        break ;
      }
      
      // Try to put an entry in the chart.
      ChartCell cell = chart[agendaItem.startOfSpan][agendaItem.spanLength - 1];
      if (cell == null) {
        cell = nbest > 1 ? new CellNBest() : new Cell1Best();
        chart[agendaItem.startOfSpan][agendaItem.spanLength - 1] = cell;
      }

      
      if (cell.add(agendaItem.parse)) {
        // If a new entry was added, count a new parse if applicable...
        
        if (agendaItem.spanLength == sentenceLength &&
            possibleRootCategories.contains(agendaItem.parse.getCategory())) {
          parsesFound++;
        }
        
        // ... and update the agenda.

        //See if any Unary Rules can be applied to the new entry.
        for (Category unaryRuleProduction : unaryRules.get(agendaItem.parse.getCategory())) {
          /* if (agendaItem.spanLength == sentenceLength) {
            break ;
          } */
          
          agenda.add(new AgendaItem(nodeFactory.makeUnary(unaryRuleProduction, agendaItem.parse, unaryRuleProduction.isForwardTypeRaised(), unaryRuleProduction.isBackwardTypeRaised()), 
              outsideProbabilitiesUpperBound[agendaItem.startOfSpan][agendaItem.startOfSpan + agendaItem.spanLength],                             
              agendaItem.startOfSpan, agendaItem.spanLength));
        }
        
        // See if the new entry can be the left argument of any binary rules.
        for (int spanLength = agendaItem.spanLength + 1; spanLength < 1 + sentenceLength - agendaItem.startOfSpan; spanLength++) {
          if (!spanConstraints.accept(agendaItem.startOfSpan, spanLength)) {
            continue;
          }
          SyntaxTreeNode leftEntry = agendaItem.parse;
          ChartCell rightCell = chart[agendaItem.startOfSpan + agendaItem.spanLength][spanLength - agendaItem.spanLength - 1];
          if (rightCell == null) continue ;
          for (SyntaxTreeNode rightEntry : rightCell.getEntries()) {
            updateAgenda(agenda, agendaItem.startOfSpan, spanLength, leftEntry, rightEntry, sentenceLength, outsideProbabilitiesUpperBound[agendaItem.startOfSpan][agendaItem.startOfSpan + spanLength]);
          }
        }
        
        // See if the new entry can be the right argument of any binary rules.
        for (int startOfSpan = 0; startOfSpan < agendaItem.startOfSpan; startOfSpan++) {
          int spanLength = agendaItem.startOfSpan + agendaItem.spanLength - startOfSpan;
          if (!spanConstraints.accept(startOfSpan, spanLength)) {
            continue;
          }
          SyntaxTreeNode rightEntry = agendaItem.parse;
          ChartCell leftCell = chart[startOfSpan][spanLength - agendaItem.spanLength - 1];
          if (leftCell == null) continue ;
          for (SyntaxTreeNode leftEntry : leftCell.getEntries()) {
            updateAgenda(agenda, startOfSpan, spanLength, leftEntry, rightEntry, sentenceLength, outsideProbabilitiesUpperBound[startOfSpan][startOfSpan + spanLength]);
          }
        }
      } 
    }

    if (parsesFound == 0) {
      // Parse failure.
      return null;
    }
    
    // Read the parses out of the final cell.
    List<SyntaxTreeNode> parses = new ArrayList<SyntaxTreeNode>();
    
    for (SyntaxTreeNode parse : chart[0][sentenceLength - 1].getEntries()) {
      // Enforce that the root node must have one of a pre-specified list of categories.
      if (possibleRootCategories.contains(parse.getCategory())) {
        parses.add(parse);
      }
    }
    
    // Sort the parses by probability.
    Collections.sort(parses);
    
    return parses;

  }

  /**
   * Computes an upper bound on the outside probabilities of a span, for use as a heuristic in A*.
   * The upper bound is simply the product of the probabilities for the most probable supertag for 
   * each word outside the span.
   */
  private double[][] computeOutsideProbabilities(List<List<SyntaxTreeNodeLeaf>> supertags)
  {
    int sentenceLength = supertags.size();
    final double[][] outsideProbability = new double[sentenceLength + 1][sentenceLength + 1];
    
    final double[] fromLeft = new double[sentenceLength + 1];
    final double[] fromRight = new double[sentenceLength + 1];

    
    fromLeft[0] = CERTAIN;
    fromRight[sentenceLength] = CERTAIN;
        
    for (int i=0 ; i<sentenceLength - 1; i++) {
      int j = sentenceLength - i;
      // The supertag list for words is sorted, so the most probably entry is at index 0.
      fromLeft[i + 1] = fromLeft[i] + supertags.get(i).get(0).probability;
      fromRight[j - 1] = fromRight[j] + supertags.get(j - 1).get(0).probability;
    }
    
    for (int i=0; i<sentenceLength+1; i++) {
      for (int j=i; j<sentenceLength + 1; j++) {
        outsideProbability[i][j] = fromLeft[i] + fromRight[j];
      }
    }
    
    return outsideProbability;
  }

  /**
   * Updates the agenda with the result of all combinators that can be applied to leftChild and rightChild.
   */
  private void updateAgenda(
      final PriorityQueue<AgendaItem> agenda, 
      final int startOfSpan,
      final int spanLength,
      final SyntaxTreeNode leftChild, 
      final SyntaxTreeNode rightChild, 
      final int sentenceLength, 
      double outsideProbabilityUpperBound)
  {

    if (!seenRules.isSeen(leftChild.getCategory(), rightChild.getCategory())) {
      return;
    }
    
    for (RuleProduction production : getRules(leftChild.getCategory(), rightChild.getCategory())) {
      // Normal form constraints from Hockenmaier and Bisk (2010).
      // Since EasyCCG does not (yet) support generalized composition with
      // degrees > 2, the formulation is simplified.
      final Combinator.RuleType leftRuleType = leftChild.getRuleType();
      final Combinator.RuleType rightRuleType = rightChild.getRuleType();
      final Combinator.RuleType resultRuleType = production.ruleType;
      
      //System.err.println(leftChild.getCategory() + " " + rightChild.getCategory());
      
      // NFC 1, forward case
      if (leftRuleType.isForward() && resultRuleType.isForward() &&
          leftRuleType.isComp() && resultRuleType.isAppOrComp1()) {
          //System.err.println("nfc1f");
        continue;
      }
      
      // NFC 1, backward case
      if (rightRuleType.isBackward() && resultRuleType.isBackward() &&
          rightRuleType.isComp() && resultRuleType.isAppOrComp1()) {
          //System.err.println("nfc1b");
        continue;
      }
      
      // NFC 2, forward case
      if (leftRuleType.isForward() && resultRuleType.isForward() &&
          leftRuleType.isComp1() && resultRuleType.isComp()) {
          //System.err.println("nfc2f");
        continue;
      }
      
      // NFC 2, backward case
      if (rightRuleType.isBackward() && resultRuleType.isBackward() &&
          rightRuleType.isComp1() && resultRuleType.isComp()) {
          //System.err.println("nfc2b");
        continue;
      }
      
      // NFC 3 is always satsified when the composition degree is max. 2.
      
      // NFC 4, forward case
      if (resultRuleType.isForward() && resultRuleType.isComp1() &&
          rightRuleType.isBackward() && rightRuleType.isComp2() &&
          leftChild.isForwardTypeRaised()) {
          //System.err.println("nfc4f");
        continue;
      }
      
      // NFC 4, backward case
      if (resultRuleType.isBackward() && resultRuleType.isComp1() &&
          leftRuleType.isForward() && leftRuleType.isComp2() &&
          rightChild.isBackwardTypeRaised()) {
          //System.err.println("nfc4b");
        continue;
      }
      
      // NFC 5, forward case
      if (resultRuleType == RuleType.FA &&
          leftChild.isForwardTypeRaised()) {
          //System.err.println("nfc5f");
        continue;
      }
      
      // NFC 5, backward case
      if (resultRuleType == RuleType.BA &&
          rightChild.isBackwardTypeRaised()) {
          //System.err.println("nfc4b");
        continue;
      }
      
      // NFC 6: not implemented. Should we?
      
      //System.err.println("ok");
      
      final AgendaItem newItem = new AgendaItem(
          nodeFactory.makeBinary(production.category, leftChild, rightChild, production.ruleType, production.headIsLeft),
          outsideProbabilityUpperBound, startOfSpan, spanLength);
      agenda.add(newItem);
    }
  }
  
  private final Map<Category, Map<Category, Collection<RuleProduction>>> ruleCache = new HashMap<Category, Map<Category, Collection<RuleProduction>>>();
  /**
   * Returns the set of binary rule productions between these two categories.
   */
  private Collection<RuleProduction> getRules(Category left, Category right) {
    Map<Category, Collection<RuleProduction>> rightToRules = ruleCache.get(left);
    if (rightToRules == null) {
      rightToRules = new HashMap<Category, Collection<RuleProduction>>();
      ruleCache.put(left, rightToRules);
    }
  
    Collection<RuleProduction> result = rightToRules.get(right);
    if (result == null) {
      result = Combinator.getRules(left, right, binaryRules);
      rightToRules.put(right, ImmutableList.copyOf(result));
    } 
    
    return result;
  }
  
  /**
   * Converts a parse into a list of supercategories.
   */
  
  private static List<Category> getSupertags(SyntaxTreeNode parse)
  {
    GetSupertagsVisitor v = new GetSupertagsVisitor();
    parse.accept(v);
    return v.result;
  }

  @Override
  public long getParsingTimeOnlyInMillis()
  {
    return parsingTimeOnly.elapsed(TimeUnit.MILLISECONDS);
  }

  @Override
  public long getTaggingTimeOnlyInMillis()
  {
    return taggingTimeOnly.elapsed(TimeUnit.MILLISECONDS);
  }
  
  private static class GetSupertagsVisitor implements SyntaxTreeNodeVisitor {
    List<Category> result = new ArrayList<Category>();

    @Override
    public void visit(SyntaxTreeNodeBinary node)
    {
      node.leftChild.accept(this);
      node.rightChild.accept(this);
    }

    @Override
    public void visit(SyntaxTreeNodeUnary node)
    {
      node.child.accept(this);
    }

    @Override
    public void visit(SyntaxTreeNodeLeaf node)
    {
      result.add(node.getCategory());
    }
  }



  /**
   * Chart Cell used for N-best parsing. It allows multiple entries with the same category, if they are not equivalent.
   */
  private class CellNBest extends ChartCell {
    private final Table<Category, Integer, SyntaxTreeNode> keyToProbability= HashBasedTable.create();
    public Collection<SyntaxTreeNode> getEntries() {
      return keyToProbability.values();
    }
    @Override
    void addEntry(Category category, int hash, SyntaxTreeNode newEntry)
    {
      keyToProbability.put(category, hash, newEntry);
    }
    @Override
    boolean isFull(Category category, int hash)
    {
      return keyToProbability.row(category).size() == nbest || keyToProbability.contains(category, hash);
    }
    @Override
    SyntaxTreeNode getEntry(Category category, int hash)
    {
      return keyToProbability.get(category, hash);
    }
  }

  /**
   * Chart Cell used for 1-best parsing.
   */
  private static class Cell1Best extends ChartCell {
    private final Map<Category, SyntaxTreeNode> keyToProbability = new HashMap<Category, SyntaxTreeNode>();
    public Collection<SyntaxTreeNode> getEntries() {
      return keyToProbability.values();
    }
    @Override
    void addEntry(Category category, int hash, SyntaxTreeNode newEntry)
    {
      keyToProbability.put(category, newEntry);
    }
    @Override
    boolean isFull(Category category, int hash)
    {
      return keyToProbability.containsKey(category);
    }
    @Override
    SyntaxTreeNode getEntry(Category category, int hash)
    {
      return keyToProbability.get(category);
    }
  }
  
  private static abstract class ChartCell {
    private double bestValue = IMPOSSIBLE;

    public ChartCell() {}
    /**
     * Possibly adds a @CellEntry to this chart cell. Returns true if the parse was added, and false if the cell was unchanged. 
     */
    public boolean add(SyntaxTreeNode entry)    
    {
      // See if the cell already has enough parses with this category.
      // All existing entries are guaranteed to have a higher probability
      if (isFull(entry.getCategory(), entry.hash)) {
        return false;
      } else {
        addEntry(entry.getCategory(), entry.hash, entry);
        
        if (entry.probability > bestValue) {
          bestValue = entry.probability;
        }
        
        return true;
      }
    }
    
    abstract boolean isFull(Category category, int hash);
    public abstract Collection<SyntaxTreeNode> getEntries();
    abstract void addEntry(Category category, int hash, SyntaxTreeNode newEntry);
    abstract SyntaxTreeNode getEntry(Category category, int hash);

  }
  

  /**
   * Handles filtering rules by CCGBank category combination.
   */
  class SeenRules {
    private Map<Category, Category> simplify = new HashMap<Category, Category>();
    private Category simplify(Category input) {
      Category result = simplify.get(input);
      if (result == null) {
        // Simplify categories for compatibility with the C&C rules file.
        result = Category.valueOf(input.toString().replaceAll("\\[X\\]", "").replaceAll("\\[nb\\]", ""));
        simplify.put(input, result);
      }

      return result;
    }

    private final boolean[][] seen;
    private final int numberOfSeenCategories;
    boolean isSeen(Category left, Category right) {
      if (seen == null) return true;
      left = simplify(left);
      right = simplify(right);
      return left.getID() < numberOfSeenCategories && right.getID() < numberOfSeenCategories && 
             seen[left.getID()][right.getID()];
    }
    
    private SeenRules(File file) throws IOException {   
      if (file == null) {
        seen = null;
        numberOfSeenCategories = 0;
      } else if (!file.exists()) {
        System.err.println("No 'seenRules' file available for model. Allowing all CCG-legal rules.");
        seen = null;
        numberOfSeenCategories = 0;
      } else {
        Table<Category, Category, Boolean> tab = HashBasedTable.create();
        int maxID = 0;
        for (String line : Util.readFile(file)) {
          // Assumes the file has the format:
          // cat1 cat2
          if (!line.startsWith("#") && !line.isEmpty()) {
            String[] fields = line.split(" ");
            Category left = Category.valueOf(fields[0]);
            Category right = Category.valueOf(fields[1]);
            maxID = Math.max(left.getID(), maxID);
            maxID = Math.max(right.getID(), maxID);
            tab.put(simplify(left), simplify(right), true);
          }
        }
        
        seen = new boolean[maxID + 1][maxID + 1];
        for (Cell<Category, Category, Boolean> entry : tab.cellSet()) {
          seen[entry.getRowKey().getID()][entry.getColumnKey().getID()] = true;
        }
        numberOfSeenCategories = seen.length;
      }
    }
  }
}