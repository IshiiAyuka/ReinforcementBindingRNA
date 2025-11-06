from Bio.Align import PairwiseAligner

global_aligner = PairwiseAligner()
global_aligner.mode = "global"
global_aligner.match_score = 1
global_aligner.mismatch_score = 0
global_aligner.open_gap_score = 0
global_aligner.extend_gap_score = 0

def global_alignment(seq1, seq2):
    score = global_aligner.score(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    return score / max_len if max_len > 0 else 0.0


local_aligner = PairwiseAligner()
local_aligner.mode = "local"
local_aligner.match_score = 1
local_aligner.mismatch_score = 0
local_aligner.open_gap_score = 0
local_aligner.extend_gap_score = 0

def local_alignment(seq1, seq2):
    score = local_aligner.score(seq1, seq2)
    # 一致スコアはアライメント長に相当。スコア自体が一致数と等しい（設定により）
    aligned_len = min(len(seq1), len(seq2))
    return score / aligned_len if aligned_len > 0 else 0.0
