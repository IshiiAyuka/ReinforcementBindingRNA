import pandas as pd
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

# CSV読み込み
df = pd.read_csv("ppi3d.csv")

# s1_sequence列を使ってFASTAファイルを作成
records = []
for i, row in df.iterrows():
    seq_id = row["subunit_1"] 
    sequence = row["s1_sequence"]
    records.append(SeqRecord(Seq(sequence), id=seq_id, description=""))

# FASTA形式で保存
SeqIO.write(records, "ppi3d_s1_sequences.fasta", "fasta")
