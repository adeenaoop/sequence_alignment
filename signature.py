def align(seq1: str, seq2: str, 
          match: int = 1, 
          mismatch: int = -1, 
          gap: int = -2,
          **kwargs) -> dict:
    """
    Returns:
        {
            'score': int,
            'aligned1': str,    # with '-' for gaps
            'aligned2': str,
            'time_ms': float,   # set by benchmark, not algorithm
        }
    """
