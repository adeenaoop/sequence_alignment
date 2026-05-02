import numpy as np

#Dynamic Programming Approach, returns optimal score and aligned sequences
#extracting the signature from signature.py

def needleman_wunsch(seq1: str, seq2: str, 
          match: int = 1, #alignment benefit 
          mismatch: int = -1, #mismatch penalty 
          gap: int = -2) -> dict: #gap penalty
    

    m = len(seq1)
    n = len(seq2)

    aligned_seq1 = ""
    aligned_seq2 = ""

    #making the matrix
    F = np.zeros((m+1,n+1), dtype = int)

    #filling first row and column
    for i in range (m+1):
        F[i][0] = i * gap
    for i in range (n+1):
        F[0][i] = i * gap

    #now filling rest of the matrix
    for i in range(1,m+1):
        for j in range(1,n+1):
            if seq1[i-1] == seq2[j-1]:
                F[i][j] = F[i-1][j-1] + 1
            else:
                diagnol = F[i-1][j-1] - 1
                up = F[i-1][j] - 2
                left = F[i][j-1] - 2
                F[i][j] = max(diagnol,up,left)

    i = m
    j = n


    #back-tracing to create aligned strings
    while i > 0 or j > 0:
        if i == 0:
            aligned_seq1 = aligned_seq1 + "_"
            aligned_seq2 = seq2[j-1] + aligned_seq2
            j -= 1
        elif j == 0:
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = "-" + aligned_seq2
            i -= 1
        else:
            if seq1[i-1] == seq2[j-1]:
                sigma = 1
            else:
                sigma = -1
            if F[i][j] == F[i-1][j-1] + sigma:
                aligned_seq1 = seq1[i-1] + aligned_seq1
                aligned_seq2 = seq2[j-1] + aligned_seq2
                i -= 1
                j -= 1
            elif F[i][j] == F[i-1][j] + gap:
                aligned_seq1 = seq1[i-1] + aligned_seq1
                aligned_seq2 = "-" + aligned_seq2
                i -= 1
            else:
                aligned_seq1 = "_" + aligned_seq1
                aligned_seq2 = seq2[j-1] + aligned_seq2
                j -= 1
                          
    for i in range(1,m+1):
        for j in range(1,n+1):
            print(f"{F[i][j]:3}", end=' ')
        print ()


    print(aligned_seq1)
    print(aligned_seq2)

needleman_wunsch("CAT", "DOG")
