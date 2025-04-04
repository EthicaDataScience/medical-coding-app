#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from tqdm import tqdm
import os

# === CONFIGURATION ===
openai.api_key = st.secrets["openai"]["api_key"]
embedding_model = "text-embedding-ada-002"
similarity_threshold = 0.85

# === AUTHENTIFICATION ===
authorized_users = ["skaba@ethicacro.com", "data.science@ethicacro.com", "data.management@ethicacro.com"]

st.title("🧬 MedDRA Automatic Coding of Adverse Events")
email = st.text_input("Enter your email address to access the application")

if email not in authorized_users:
    st.warning("You are not authorized to access this application.")
    st.stop()

# === CHARGEMENT FICHIER MedDRA PRÉDÉFINI ===
meddra_path = "meddra_data.xlsx"  # Le fichier doit être dans le répertoire du projet
df_meddra = pd.read_excel(meddra_path)

# === UPLOAD DU FICHIER AE SEUL ===
uploaded_ae = st.file_uploader("Upload AE file (.txt with tabulation)", type="txt")

if uploaded_ae:
    df_ae = pd.read_csv(uploaded_ae, sep="\t", encoding="utf-8")

    # === FONCTION POUR GÉNÉRER DES EMBEDDINGS ===
    def get_embedding(text, model=embedding_model):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model).data[0].embedding

    # === PRÉCALCUL DES EMBEDDINGS POUR MedDRA ===
    st.info("Calculating embeddings for MedDRA reference terms...")
    meddra_embeddings = []
    for _, row in tqdm(df_meddra.iterrows(), total=len(df_meddra)):
        for col in ["llt_name", "pt_name"]:
            if pd.notna(row[col]):
                embedding = get_embedding(row[col])
                meddra_embeddings.append({
                    "source": col,
                    "term": row[col],
                    "embedding": embedding,
                    "row": row
                })

    # === TRAITEMENT DES AETERM ===
    st.info("Searching for semantic matches...")
    ae_coding_meddra = []

    for _, ae_row in tqdm(df_ae.iterrows(), total=len(df_ae)):
        ae_term = ae_row["AETERM"]
        if pd.isna(ae_term) or ae_term.strip() == "":
            continue

        ae_embedding = get_embedding(ae_term)

        max_sim = 0
        best_match = None
        for entry in meddra_embeddings:
            sim = cosine_similarity([ae_embedding], [entry["embedding"]])[0][0]
            if sim > max_sim:
                max_sim = sim
                best_match = entry

        combined_row = ae_row.to_dict()
        if best_match:
            matched_row = best_match["row"]
            for col in df_meddra.columns:
                combined_row[col] = matched_row.get(col, None)
            combined_row.update({
                "matched_term": best_match["term"],
                "matched_source": best_match["source"],
                "similarity": max_sim
            })
        else:
            combined_row.update({
                "matched_term": None,
                "matched_source": None,
                "similarity": max_sim
            })
            for col in df_meddra.columns:
                combined_row[col] = None

        ae_coding_meddra.append(combined_row)

    df_result = pd.DataFrame(ae_coding_meddra)
    st.success("✅ Codage terminé avec succès !")
    st.dataframe(df_result.head(20))

    # === EXPORT FINAL ===
    output = BytesIO()
    df_result.to_excel(output, index=False, engine='openpyxl')
    st.download_button(
        "📥 Download encoded file",
        data=output.getvalue(),
        file_name="AE_CODING_MEDDRA.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
