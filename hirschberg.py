
import sys
sys.setrecursionlimit(100000)

def hirschberg(seq1: str, seq2: str,
               match: int = 1,
               mismatch: int = -1,
               gap: int = -2) -> dict:
    """
    Space-efficient global alignment via Hirschberg's divide & conquer.
    Returns the same optimal alignment as Needleman-Wunsch, but uses
    O(min(m, n)) space instead of O(mn).
    """
    
    def nw_score(s1: str, s2: str) -> list:
        """
        Compute only the LAST ROW of the Needleman-Wunsch DP matrix.
        Uses O(len(s2)) space by keeping only two rows at a time.
        Returns: list of length len(s2)+1 with the final row scores.
        """
        n = len(s2)
        prev = [j * gap for j in range(n + 1)]   # row 0
        curr = [0] * (n + 1)
        
        for i in range(1, len(s1) + 1):
            curr[0] = i * gap
            for j in range(1, n + 1):
                sigma = match if s1[i-1] == s2[j-1] else mismatch
                curr[j] = max(
                    prev[j-1] + sigma,    # diagonal
                    prev[j]   + gap,      # up
                    curr[j-1] + gap       # left
                )
            prev, curr = curr, prev       # swap rows, reuse memory
        
        return prev   # this is the last row
    
    def needleman_wunsch_full(s1: str, s2: str) -> tuple:
        """
        Base case: standard NW with full matrix and traceback.
        Used when one sequence is short enough that recursion isn't needed.
        Returns (aligned1, aligned2).
        """
        m, n = len(s1), len(s2)
        F = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            F[i][0] = i * gap
        for j in range(n + 1):
            F[0][j] = j * gap
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                sigma = match if s1[i-1] == s2[j-1] else mismatch
                F[i][j] = max(
                    F[i-1][j-1] + sigma,
                    F[i-1][j]   + gap,
                    F[i][j-1]   + gap
                )
        
        # Traceback
        a1, a2 = [], []
        i, j = m, n
        while i > 0 or j > 0:
            if i == 0:
                a1.append('-'); a2.append(s2[j-1]); j -= 1
            elif j == 0:
                a1.append(s1[i-1]); a2.append('-'); i -= 1
            else:
                sigma = match if s1[i-1] == s2[j-1] else mismatch
                if F[i][j] == F[i-1][j-1] + sigma:
                    a1.append(s1[i-1]); a2.append(s2[j-1])
                    i -= 1; j -= 1
                elif F[i][j] == F[i-1][j] + gap:
                    a1.append(s1[i-1]); a2.append('-')
                    i -= 1
                else:
                    a1.append('-'); a2.append(s2[j-1])
                    j -= 1
        
        return ''.join(reversed(a1)), ''.join(reversed(a2))
    
    def hirschberg_recursive(s1: str, s2: str) -> tuple:
        """
        The recursive heart of Hirschberg's algorithm.
        Returns (aligned1, aligned2) for the optimal global alignment.
        """
        m, n = len(s1), len(s2)
        
        # Base cases
        if m == 0:
            # s1 empty: align s2 against all gaps
            return '-' * n, s2
        if n == 0:
            # s2 empty: align s1 against all gaps
            return s1, '-' * m
        if m == 1 or n == 1:
            # One sequence too small to split — fall back to standard NW
            return needleman_wunsch_full(s1, s2)
        
        # Divide: split s1 at the middle row
        mid = m // 2
        
        # Forward pass on top half: NW on s1[:mid] vs s2
        scoreL = nw_score(s1[:mid], s2)
        
        # Backward pass on bottom half: NW on reversed s1[mid:] vs reversed s2
        scoreR = nw_score(s1[mid:][::-1], s2[::-1])
        
        # Find the column j* where the optimal alignment crosses the middle row
        # j* maximizes scoreL[j] + scoreR[n - j]
        best_j = 0
        best_sum = scoreL[0] + scoreR[n]
        for j in range(1, n + 1):
            total = scoreL[j] + scoreR[n - j]
            if total > best_sum:
                best_sum = total
                best_j = j
        
        # Conquer: recursively align the two halves
        left1,  left2  = hirschberg_recursive(s1[:mid],  s2[:best_j])
        right1, right2 = hirschberg_recursive(s1[mid:],  s2[best_j:])
        
        # Combine: concatenate the two alignments
        return left1 + right1, left2 + right2
    
    aligned1, aligned2 = hirschberg_recursive(seq1, seq2)
    
    # Compute the final score from the alignment
    score = 0
    for c1, c2 in zip(aligned1, aligned2):
        if c1 == '-' or c2 == '-':
            score += gap
        elif c1 == c2:
            score += match
        else:
            score += mismatch
    
    return {
        'score': score,
        'aligned1': aligned1,
        'aligned2': aligned2,
    }


if __name__ == "__main__":
    # Test 1: same example as NW for verification
    result = hirschberg("ACGT", "AGCT")
    print(f"Test 1 — Score: {result['score']}")
    print(result['aligned1'])
    print(result['aligned2'])
    print()
    
    # Test 2: longer sequences to show it actually works
    result = hirschberg("AGTACGCA", "TATGC")
    print(f"Test 2 — Score: {result['score']}")
    print(result['aligned1'])
    print(result['aligned2'])
