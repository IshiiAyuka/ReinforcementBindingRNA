import os
import numpy as np
import pandas as pd
from ZHMolGraph import ZHMolGraph
import pickle as pkl
import argparse
import csv

def get_seq_from_csv():
    df = pd.read_csv("predictions_trimmed.csv")
    protein = str(df["protein_seq"].iloc[0]).strip()
    rna     = str(df["pred_rna_seq"].iloc[0]).strip()
    return protein, rna

def main():
    protein, rna = get_seq_from_csv()

    print(f"Input RNA sequence: {rna}")
    print(f"Input protein sequence: {protein}")

    RNA_seq_df = pd.DataFrame([{'RNA_aa_code': rna}])
    protein_seq_df = pd.DataFrame([{'target_aa_code': protein}])


    model_Dataset = 'NPInter2'
    with open('data/data/Mol2Vec/RPI_' + model_Dataset + '_rnafm_embed_normal.pkl', 'rb') as file:
        rnas = pkl.load(file)

    with open('data/data/Mol2Vec/RPI_' + model_Dataset + '_proteinprottrans_embed_normal.pkl', 'rb') as file:
        proteins = pkl.load(file)

    vecnn_object = ZHMolGraph.ZHMolGraph(
        interactions_location=f'data/data/interactions/dataset_RPI_{model_Dataset}_RP.csv',
        interactions=None,
        interaction_y_name='Y',
        rnas_dataframe=rnas,
        rna_seq_name='RNA_aa_code',
        proteins_dataframe=proteins,
        protein_seq_name='target_aa_code',
        model_out_dir=f'trained_model/trained_model/ZHMolGraph_VecNN_model_RPI_{model_Dataset}/',
        debug=False
    )
    rnas_embeddings_array = np.array(rnas['normalized_embeddings'].tolist())
    vecnn_object.mean_rna_embeddings = np.mean(rnas_embeddings_array, axis=0)
    vecnn_object.centered_rna_embeddings = rnas_embeddings_array - vecnn_object.mean_rna_embeddings
    vecnn_object.centered_rna_embeddings_length = np.mean(
        np.sqrt(np.sum(vecnn_object.centered_rna_embeddings * vecnn_object.centered_rna_embeddings, axis=1))
    )

    proteins_embeddings_array = np.array(proteins['normalized_embeddings'].tolist())
    vecnn_object.mean_protein_embeddings = np.mean(proteins_embeddings_array, axis=0)
    vecnn_object.centered_protein_embeddings = proteins_embeddings_array - vecnn_object.mean_protein_embeddings
    vecnn_object.centered_protein_embeddings_length = np.mean(
        np.sqrt(np.sum(vecnn_object.centered_protein_embeddings * vecnn_object.centered_protein_embeddings, axis=1))
    )

    test_rna = vecnn_object.get_rnafm_embeddings(prediction_interactions=RNA_seq_df,
                                                 replace_dataframe=False,
                                                 return_normalisation_conststants=True)
    test_protein = vecnn_object.get_ProtTrans_embeddings(prediction_interactions=protein_seq_df,
                                                         replace_dataframe=False,
                                                         return_normalisation_conststants=True)

    vecnn_object.rnas_dataframe = test_rna
    vecnn_object.rna_list = list(vecnn_object.rnas_dataframe[vecnn_object.rna_seq_name])
    vecnn_object.proteins_dataframe = test_protein
    vecnn_object.protein_list = list(vecnn_object.proteins_dataframe[vecnn_object.protein_seq_name])

    interactions_seqpairs = pd.concat([RNA_seq_df, protein_seq_df], axis=1)

    vecnn_object.predict_RPI(model_dataset=model_Dataset,
                             graphsage_path=vecnn_object.model_out_dir,
                             jobname="test",
                             test_dataframe=interactions_seqpairs,
                             rna_vector_length=640,
                             protein_vector_length=1024,
                             rnas=test_rna,
                             proteins=test_protein,
                             embedding_type='Pretrain',
                             graphsage_embedding=1)

    scores = vecnn_object.averaged_results
    average_score = sum(scores) / len(scores)
    print(f"score:{average_score:3f}")

if __name__ == "__main__":
    main()
