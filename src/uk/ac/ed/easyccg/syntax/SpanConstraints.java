/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package uk.ac.ed.easyccg.syntax;

import java.util.ArrayList;
import java.util.List;

/**
 * A SpanConstraints object represents a set of constraints imposed on the
 * parse of a sentence. Each span constraint is a span that <b>should be
 * present</b> in the parse as a constituent.
 * @author Kilian Evang
 */
class SpanConstraints {
  
  private final List<int[]> constraints = new ArrayList<>();
  
  /**
   * Adds a new span constraint.
   * @param startOfSpan start offset of the constraint
   * @param spanLength length of the constraint
   */
  public void addConstraint(final int startOfSpan, final int spanLength) {
    constraints.add(new int[] {startOfSpan, spanLength});
  }
  
  /**
   * Takes a proposed span and rejects it if it properly overlaps with one of
   * the span constraints (which means that they cannot coexist in one parse).
   * Otherwise accepts it.
   * @param startOfSpan
   * @param spanLength
   * @return false if rejected, true if accepted
   */
  public boolean accept(final int startOfSpan, final int spanLength) {
    final int endOfSpan = startOfSpan + spanLength;
    for (int[] constraint : constraints) {
      final int startOfConstraint = constraint[0];
      final int constraintLength = constraint[1];
      final int endOfConstraint = startOfConstraint + constraintLength;
      // Return false if span properly overlaps with constraint:
      if (startOfSpan < startOfConstraint
              && endOfSpan > startOfConstraint
              && endOfSpan < endOfConstraint) {
        return false;
      }
      if (startOfSpan > startOfConstraint
              && startOfSpan < endOfConstraint
              && endOfSpan > endOfConstraint) {
        return false;
      }
    }
    return true;
  }
  
}
