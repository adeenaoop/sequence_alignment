import numpy as np

# Constrained Dynamic Programming Approach, computes global alignment
# but only fills cells within a diagonal band of width k (|i-j| <= k)
# Optimal only when the true alignment path lies within the band
def banded_dp(seq1: str, seq2: str,
              k: int = 3,         # half-bandwidth: only cells where |i-j| <= k are filled
              match: int = 1,     # alignment benefit
              mismatch: int = -1, # mismatch penalty
              gap: int = -2):     # gap penalty

    m = len(seq1)
    n = len(seq2)
    aligned_seq1 = ""
    aligned_seq2 = ""

    # if sequences differ in length by more than k,
    # the alignment path must exit the band — no valid alignment possible
    if abs(m - n) > k:
        print("Band too narrow: |len(seq1) - len(seq2)| exceeds k.")
        print("Increase k to at least", abs(m - n))
        return

    # making the matrix, fill with -infinity (unreachable by default)
    NEG_INF = -999
    F = np.full((m + 1, n + 1), NEG_INF, dtype=int)

    # base cases — only initialise cells within the band
    F[0][0] = 0
    for i in range(1, m + 1):
        if abs(i - 0) <= k:
            F[i][0] = i * gap
    for j in range(1, n + 1):
        if abs(0 - j) <= k:
            F[0][j] = j * gap

    # filling the matrix — only within the band |i-j| <= k
    for i in range(1, m + 1):
        j_lo = max(1, i - k)
        j_hi = min(n, i + k)
        for j in range(j_lo, j_hi + 1):
            if seq1[i - 1] == seq2[j - 1]:
                sigma = match
            else:
                sigma = mismatch
            diagonal = F[i-1][j-1] + sigma  if F[i-1][j-1] != NEG_INF else NEG_INF
            up       = F[i-1][j]   + gap    if F[i-1][j]   != NEG_INF else NEG_INF
            left     = F[i][j-1]   + gap    if F[i][j-1]   != NEG_INF else NEG_INF
            F[i][j]  = max(diagonal, up, left)

    # back-tracing to create aligned strings
    i = m
    j = n
    while i > 0 or j > 0:
        if i == 0:
            aligned_seq1 = "_" + aligned_seq1
            aligned_seq2 = seq2[j - 1] + aligned_seq2
            j -= 1
        elif j == 0:
            aligned_seq1 = seq1[i - 1] + aligned_seq1
            aligned_seq2 = "-" + aligned_seq2
            i -= 1
        else:
            if seq1[i - 1] == seq2[j - 1]:
                sigma = match
            else:
                sigma = mismatch
            if F[i][j] == F[i-1][j-1] + sigma:
                aligned_seq1 = seq1[i - 1] + aligned_seq1
                aligned_seq2 = seq2[j - 1] + aligned_seq2
                i -= 1
                j -= 1
            elif F[i][j] == F[i-1][j] + gap:
                aligned_seq1 = seq1[i - 1] + aligned_seq1
                aligned_seq2 = "-" + aligned_seq2
                i -= 1
            else:
                aligned_seq1 = "_" + aligned_seq1
                aligned_seq2 = seq2[j - 1] + aligned_seq2
                j -= 1

    # printing the band (dots for out-of-band cells)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if F[i][j] == NEG_INF:
                print(f"{'.'  :>4}", end=' ')
            else:
                print(f"{F[i][j]:3}", end=' ')
        print()
    print(aligned_seq1)
    print(aligned_seq2)

banded_dp("ATCGTA", "ATCGAA", k=2)
