1. タンパク質配列入力(PPI3Dのprotein-nucleic acid, 配列長は1022以下)
2. ESM2で特徴量抽出(t30-150Mの学習済み重みを使用, ptファイルを出力)
3. デコーダにタンパク質特徴量を入力(PPI3Dの対応するRNA配列を正解配列として学習)
4. デコーダがRNA配列を出力(10~100nt)
5. タンパク質配列を強化学習モジュールに入力(SwissProt(Molecular functionがRNA-binding)を使用)

データセットは、取得したままの状態で使用


#!/bin/bash
#$ -l h_vmem=100G

conda activate [実行する仮想環境]

cd [ファイルのあるディレクトリ]

nohup python -u [実行するファイルのタイトル].py > output.log 2> error.log &
