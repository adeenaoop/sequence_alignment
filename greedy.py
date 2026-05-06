def greedy_sequence_alignment(seq1: str, seq2: str,
                               match: int = 1,
                               mismatch: int = -1,
                               gap: int = -2) -> dict:
    """
    Greedy Heuristic for pairwise sequence alignment.
    Strategy: repeatedly find the longest common substring (LCS),
    align it, then recurse on the remaining left/right flanks.
    Time: O(n log n) approx | Space: O(n)
    """

    def find_longest_common_substring(s1, s2):
        """Returns (len, start_in_s1, start_in_s2) of the longest common substring."""
        best_len = 0
        best_i = 0
        best_j = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                l = 0
                while i + l < len(s1) and j + l < len(s2) and s1[i + l] == s2[j + l]:
                    l += 1
                if l > best_len:
                    best_len = l
                    best_i = i
                    best_j = j
        return best_len, best_i, best_j

    def align_greedy(s1, s2):
        """Recursively aligns s1 and s2 using greedy LCS anchoring."""

        # Base cases
        if len(s1) == 0:
            return "_" * len(s2), s2
        if len(s2) == 0:
            return s1, "-" * len(s1)

        lcs_len, i, j = find_longest_common_substring(s1, s2)

        # No common substring found — align everything with gaps
        if lcs_len == 0:
            if len(s1) >= len(s2):
                pad = len(s1) - len(s2)
                return s1, s2 + "-" * pad
            else:
                pad = len(s2) - len(s1)
                return "_" * pad + s1, s2

        # Anchor: the matched block
        anchor1 = s1[i: i + lcs_len]
        anchor2 = s2[j: j + lcs_len]  # identical to anchor1

        # Recurse on left flanks
        left1, left2 = align_greedy(s1[:i], s2[:j])

        # Recurse on right flanks
        right1, right2 = align_greedy(s1[i + lcs_len:], s2[j + lcs_len:])

        aligned1 = left1 + anchor1 + right1
        aligned2 = left2 + anchor2 + right2

        return aligned1, aligned2

    def compute_score(a1, a2, match, mismatch, gap):
        score = 0
        for c1, c2 in zip(a1, a2):
            if c1 == "_" or c2 == "-":
                score += gap
            elif c1 == c2:
                score += match
            else:
                score += mismatch
        return score

    aligned_seq1, aligned_seq2 = align_greedy(seq1, seq2)
    score = compute_score(aligned_seq1, aligned_seq2, match, mismatch, gap)

    # Pretty print
    print(f"Sequence 1: {seq1}")
    print(f"Sequence 2: {seq2}\n")
    print(f"Aligned 1:  {aligned_seq1}")
    print(f"Aligned 2:  {aligned_seq2}")
    print(f"Score:      {score}")

    return {
        "aligned_seq1": aligned_seq1,
        "aligned_seq2": aligned_seq2,
        "score": score
    }


# --- Test cases matching your milestone benchmarks ---
print("=== Test 1: Simple ===")
greedy_sequence_alignment("CAT", "DOG")

print("\n=== Test 2: Similar sequences ===")
greedy_sequence_alignment("AGCTAGCT", "AGCAAGCT")

print("\n=== Test 3: DNA-style ===")
greedy_sequence_alignment("ACGTACGT", "ACGTTTACGT")
