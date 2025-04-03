#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import os

# === CONFIGURATION ===
openai.api_key = st.secrets["openai"]["api_key"]
embedding_model = "text-embedding-ada-002"
similarity_threshold = 0.85

# === FONCTION POUR GÉNÉRER DES EMBEDDINGS ===
def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

# === AUTHENTIFICATION ===
authorized_users = ["skaba@ethicacro.com", "data.science@ethicacro.com", "data.management@ethicacro.com"]

st.title("🧬 Codage Automatique MedDRA des Effets Indésirables")

email = st.text_input("Entrez votre adresse email pour accéder à l'application")

if email in authorized_users:
    st.success("Accès autorisé. Vous pouvez maintenant charger vos fichiers.")

    ae_file = st.file_uploader("Charger le fichier AE (.txt)", type=["txt"])
    meddra_file = st.file_uploader("Charger le fichier MedDRA (.xlsx)", type=["xlsx"])

    if ae_file and meddra_file:
        df_ae = pd.read_csv(ae_file, sep="\t", encoding="utf-8")
        df_meddra = pd.read_excel(meddra_file)

        st.info("Génération des embeddings pour la table MedDRA...")
        meddra_embeddings = []
        for _, row in df_meddra.iterrows():
            for col in ["llt_name", "pt_name"]:
                if pd.notna(row[col]):
                    embedding = get_embedding(row[col])
                    meddra_embeddings.append({
                        "source": col,
                        "term": row[col],
                        "embedding": embedding,
                        "row": row
                    })

        st.info("Recherche des correspondances sémantiques avec les AETERM...")
        ae_coding_meddra = []

        for _, ae_row in df_ae.iterrows():
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

            if max_sim >= similarity_threshold:
                matched_row = best_match["row"]
                combined_row = ae_row.to_dict()
                combined_row.update({
                    "matched_term": best_match["term"],
                    "matched_source": best_match["source"],
                    "matched_meddra_code": matched_row.get("code", None),
                    "matched_preferred_term": matched_row.get("pt_name", None),
                    "similarity": max_sim
                })
                ae_coding_meddra.append(combined_row)

        df_result = pd.DataFrame(ae_coding_meddra)
        st.success("✅ Correspondance terminée. Vous pouvez télécharger le fichier résultat ci-dessous.")
        st.dataframe(df_result.head(20))

        # === EXPORT ===
        st.download_button(
            label="Télécharger le fichier AE_CODING_MEDDRA.xlsx",
            data=df_result.to_excel("AE_CODING_MEDDRA.xlsx", index=False, engine='openpyxl'),
            file_name="AE_CODING_MEDDRA.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.error("Accès refusé. Votre adresse email n'est pas autorisée.")


# In[ ]:




