
import numpy as np

#Dynamic Programming Approach, Finds the best matching sub-region of seq1 and seq2
#extracting the signature from signature.py

def smith_waterman(seq1: str, seq2: str, 
          match: int = 1, #alignment benefit 
          mismatch: int = -1, #mismatch penalty 
          gap: int = -2) -> dict: #gap penalty
    

    m = len(seq1)
    n = len(seq2)

    aligned_seq1 = ""
    aligned_seq2 = ""

    #making the matrix
    F = np.zeros((m+1,n+1), dtype = int)

    #its important to note that the first row and column remain 0

    max_score = 0
    max_pos = (0,0)

    #now filling rest of the matrix
    for i in range(1,m+1):
        for j in range(1,n+1):
            if seq1[i-1] == seq2[j-1]:
                F[i][j] = F[i-1][j-1] + 1
            else:
                diagnol = F[i-1][j-1] - 1
                up = F[i-1][j] - 2
                left = F[i][j-1] - 2
                F[i][j] = max(0,diagnol,up,left) #0 means we are resetting the alignment here and will start fresh now
            
            if F[i][j] > max_score:
                max_score = F[i][j]
                max_pos = (i,j)

    i, j = max_pos

    #we start from the max cell and stop when it reaches 0
    #back-tracing to create aligned strings
    while i > 0 and j > 0 and F[i][j]>0:
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

smith_waterman("XXACGTYY", "ZZACGTWW")
