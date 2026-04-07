import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Diagnostic énergétique des bâtiments",
    page_icon="🏠",
    layout="wide"
)


# =========================
# OUTILS
# =========================
def existing_cols(df, candidates):
    return [c for c in candidates if c in df.columns]


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def format_pct(x):
    return f"{x:.1f} %"


def safe_mode(series):
    s = series.dropna()
    if s.empty:
        return "Non disponible"
    return s.mode().iloc[0]


def decode_energy_label(series):
    """
    Convertit une colonne DPE/GES éventuellement encodée en 0..6 vers A..G.
    Si la colonne contient déjà A..G, elle est conservée.
    """
    mapping = {
        0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G",
        0.0: "A", 1.0: "B", 2.0: "C", 3.0: "D", 4.0: "E", 5.0: "F", 6.0: "G",
        "0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G",
        "0.0": "A", "1.0": "B", "2.0": "C", "3.0": "D", "4.0": "E", "5.0": "F", "6.0": "G",
        "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G"
    }
    return series.astype(str).str.strip().map(mapping).fillna(series.astype(str).str.strip())


def build_ab_fg_comparison(df, dpe_col, conso_col, ges_num_col, surf_col, period_col, isolation_col):
    fg = df[df[dpe_col].isin(["F", "G"])].copy()
    ab = df[df[dpe_col].isin(["A", "B"])].copy()

    rows = []
    for label, sub in [("Passoires thermiques (F-G)", fg), ("Bâtiments performants (A-B)", ab)]:
        rows.append({
            "Groupe": label,
            "Nombre de bâtiments": len(sub),
            "Surface moyenne": round(sub[surf_col].mean(), 2) if surf_col else np.nan,
            "Consommation moyenne": round(sub[conso_col].mean(), 2) if conso_col else np.nan,
            "GES moyen": round(sub[ges_num_col].mean(), 2) if ges_num_col else np.nan,
            "Période dominante": safe_mode(sub[period_col]) if period_col else "Non disponible",
            "Isolation enveloppe dominante": safe_mode(sub[isolation_col]) if isolation_col else "Non disponible",
        })
    return pd.DataFrame(rows)


def train_rf_model(df, target_col, numeric_features, categorical_features):
    features = numeric_features + categorical_features
    tmp = df[features + [target_col]].copy()
    tmp = tmp[tmp[target_col].notna()].copy()

    if tmp.empty or tmp[target_col].nunique() < 2:
        return None, None

    X = tmp[features]
    y = tmp[target_col].astype(str)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc


def get_feature_importance_table(model):
    try:
        preprocessor = model.named_steps["preprocessor"]
        rf = model.named_steps["rf"]

        feature_names = preprocessor.get_feature_names_out()
        importances = rf.feature_importances_

        imp = pd.DataFrame({
            "Variable": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        return imp
    except Exception:
        return pd.DataFrame(columns=["Variable", "Importance"])


def build_pca_dataframe(df, dpe_col):
    preferred_numeric = [
        "Conso_5_usages_é_finale",
        "Conso_5_usages/m²_é_finale",
        "Conso_5_usages_é_primaire",
        "Conso_5_usages_par_m²_é_primaire",
        "Emission_GES_5_usages",
        "Emission_GES_5_usages_par_m²",
        "Surface_habitable_logement",
        "Hauteur_sous-plafond",
        "Ubat_W/m²_K",
        "Besoin_chauffage",
        "Déperditions_murs",
        "Deperditions_enveloppe",
        "Déperditions_renouvellement_air"
    ]

    numeric_cols = existing_cols(df, preferred_numeric)
    if len(numeric_cols) < 4:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in numeric_cols if df[c].notna().sum() > 0][:12]

    if len(numeric_cols) < 2:
        return None, None, None

    pca_df = df[numeric_cols].copy()
    pca_df = pca_df.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X = imputer.fit_transform(pca_df)
    X = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    out = pd.DataFrame({
        "Dim1": coords[:, 0],
        "Dim2": coords[:, 1]
    })

    if dpe_col in df.columns:
        out[dpe_col] = df[dpe_col].values

    return out, pca, numeric_cols


periode_labels = {
    0: "avant 1948",
    1: "1948-1974",
    2: "1975-1977",
    3: "1978-1982",
    4: "1983-1988",
    5: "1989-2000",
    6: "2001-2005",
    7: "2006-2012",
    8: "2013-2021",
    9: "après 2021"
}

qualite_labels = {
    0: "insuffisante",
    1: "moyenne",
    2: "bonne",
    3: "très bonne"
}


# =========================
# CHARGEMENT
# =========================
st.title("🏠 Diagnostic énergétique des bâtiments")
st.markdown(
    """
Cette application interactive met en valeur un projet d’analyse des **DPE des bâtiments**.  
Elle permet de :
- visualiser les tendances globales,
- repérer les **passoires thermiques**,
- comparer les profils énergétiques,
- proposer un **diagnostic prédictif simplifié**.
"""
)

with st.sidebar:
    st.header("Chargement des données")
    st.markdown("**Fichier chargé automatiquement :** `df_final.csv`")

df = pd.read_csv("df_final.csv")

# Colonnes clés
dpe_col = first_existing(df, ["Etiquette_DPE"])
ges_col = first_existing(df, ["Etiquette_GES"])
conso_col = first_existing(df, ["Conso_5_usages_é_finale"])
surf_col = first_existing(df, ["Surface_habitable_logement"])
period_col = first_existing(df, ["Période_construction"])
isolation_col = first_existing(df, ["Qualité_isolation_enveloppe"])
ges_num_col = first_existing(df, ["Emission_GES_5_usages"])

# Décodage DPE/GES
if dpe_col:
    df[dpe_col] = decode_energy_label(df[dpe_col])

if ges_col:
    df[ges_col] = decode_energy_label(df[ges_col])

# =========================
# FILTRES
# =========================
with st.sidebar:
    st.header("Filtres")

    df_filtered = df.copy()

    if period_col:
        choices = sorted(df[period_col].dropna().astype(str).unique().tolist())
        selected = st.multiselect("Période de construction", choices, default=choices)
        if selected:
            df_filtered = df_filtered[df_filtered[period_col].astype(str).isin(selected)]

    if dpe_col:
        choices = [x for x in list("ABCDEFG") if x in df[dpe_col].astype(str).unique().tolist()]
        selected = st.multiselect("Étiquette DPE", choices, default=choices)
        if selected:
            df_filtered = df_filtered[df_filtered[dpe_col].astype(str).isin(selected)]

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "Vue d’ensemble",
    "Passoires thermiques",
    "ACP / profils",
    "Diagnostic prédictif"
])

# =========================
# TAB 1
# =========================
with tab1:
    st.subheader("Vue d'ensemble du parc étudié")

    col1, col2, col3, col4 = st.columns(4)

    nb_bat = len(df_filtered)
    part_fg = (df_filtered[dpe_col].isin(["F", "G"]).mean() * 100) if dpe_col and nb_bat > 0 else np.nan
    conso_mean = df_filtered[conso_col].mean() if conso_col else np.nan
    ges_mean = df_filtered[ges_num_col].mean() if ges_num_col else np.nan

    col1.metric("Nombre de bâtiments", f"{nb_bat:,}".replace(",", " "))
    col2.metric("Part de passoires thermiques", format_pct(part_fg) if not pd.isna(part_fg) else "N/A")
    col3.metric("Consommation moyenne", f"{conso_mean:,.1f}".replace(",", " ") if not pd.isna(conso_mean) else "N/A")
    col4.metric("GES moyen", f"{ges_mean:,.1f}".replace(",", " ") if not pd.isna(ges_mean) else "N/A")

    c1, c2 = st.columns(2)

    with c1:
        if dpe_col:
            dpe_counts = (
                df_filtered[dpe_col]
                .value_counts()
                .reindex(list("ABCDEFG"), fill_value=0)
                .reset_index()
            )
            dpe_counts.columns = ["Etiquette_DPE", "Nombre"]
            fig = px.bar(
                dpe_counts,
                x="Etiquette_DPE",
                y="Nombre",
                title="Répartition des classes DPE"
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if ges_col:
            ges_counts = (
                df_filtered[ges_col]
                .value_counts()
                .reindex(list("ABCDEFG"), fill_value=0)
                .reset_index()
            )
            ges_counts.columns = ["Etiquette_GES", "Nombre"]
            fig = px.bar(
                ges_counts,
                x="Etiquette_GES",
                y="Nombre",
                title="Répartition des classes GES"
            )
            st.plotly_chart(fig, use_container_width=True)

    if conso_col and dpe_col:
        fig = px.box(
            df_filtered,
            x=dpe_col,
            y=conso_col,
            category_orders={dpe_col: list("ABCDEFG")},
            title="Consommation finale selon l'étiquette DPE"
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2
# =========================
with tab2:
    st.subheader("Analyse des passoires thermiques")

    if dpe_col is None:
        st.warning("La colonne Etiquette_DPE est nécessaire pour cette section.")
    else:
        comparison_df = build_ab_fg_comparison(
            df_filtered, dpe_col, conso_col, ges_num_col, surf_col, period_col, isolation_col
        )
        st.dataframe(comparison_df, use_container_width=True)

        st.markdown(
            """
Cette comparaison oppose les bâtiments les plus performants (**A-B**) aux plus énergivores (**F-G**).  
Elle permet de faire ressortir les grands profils énergétiques présents dans les données.
"""
        )

        if period_col:
            fg = df_filtered[df_filtered[dpe_col].isin(["F", "G"])]
            if not fg.empty:
                counts = fg[period_col].value_counts().reset_index()
                counts.columns = ["Code", "Nombre"]

                counts["Libellé"] = counts["Code"].map(periode_labels)

                fig = px.bar(
                    counts,
                    x="Code",
                    y="Nombre",
                    text="Libellé",
                    title="Périodes de construction les plus fréquentes parmi les passoires thermiques"
                )

                fig.update_traces(
                    textposition="outside",
                    hovertemplate="<b>Code :</b> %{x}<br><b>Période :</b> %{text}<br><b>Nombre :</b> %{y}<extra></extra>"
                )

                fig.update_xaxes(
                    tickmode="array",
                    tickvals=counts["Code"],
                    ticktext=counts["Code"]
                )

                fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")

                st.plotly_chart(fig, use_container_width=True)

        if isolation_col:
            fg = df_filtered[df_filtered[dpe_col].isin(["F", "G"])]
            if not fg.empty:
                counts = fg[isolation_col].value_counts().reset_index()
                counts.columns = ["Code", "Nombre"]

                counts["Libellé"] = counts["Code"].map(qualite_labels)

                fig = px.bar(
                    counts,
                    x="Code",
                    y="Nombre",
                    text="Libellé",
                    title="Qualité d'isolation dominante parmi les passoires thermiques"
                )

                fig.update_traces(
                    textposition="outside",
                    hovertemplate="<b>Code :</b> %{x}<br><b>Qualité :</b> %{text}<br><b>Nombre :</b> %{y}<extra></extra>"
                )

                fig.update_xaxes(
                    tickmode="array",
                    tickvals=counts["Code"],
                    ticktext=counts["Code"]
                )

                fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")

                st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3
# =========================
with tab3:
    st.subheader("Projection ACP des profils énergétiques")

    pca_df, pca_model, numeric_used = build_pca_dataframe(df_filtered, dpe_col)

    if pca_df is None:
        st.warning("Pas assez de variables numériques disponibles pour construire l'ACP.")
    else:
        options_color = [opt for opt in [dpe_col, period_col, isolation_col] if opt is not None]
        color_choice = st.selectbox("Colorer les points selon", options=options_color)

        if color_choice and color_choice in df_filtered.columns:
            pca_df[color_choice] = df_filtered[color_choice].astype(str).fillna("NA").values

        fig = px.scatter(
            pca_df,
            x="Dim1",
            y="Dim2",
            color=color_choice if color_choice in pca_df.columns else None,
            title="ACP — représentation des bâtiments dans un espace réduit à 2 dimensions",
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)

        var1 = pca_model.explained_variance_ratio_[0] * 100
        var2 = pca_model.explained_variance_ratio_[1] * 100

        st.markdown(
            f"""
- **Variance expliquée par l’axe 1 :** {var1:.2f} %
- **Variance expliquée par l’axe 2 :** {var2:.2f} %

Cette projection permet de voir si certains profils énergétiques ont tendance à se regrouper ou à se distinguer.
"""
        )

        with st.expander("Variables utilisées pour l’ACP"):
            st.write(numeric_used)

# =========================
# TAB 4
# =========================
with tab4:
    st.subheader("Diagnostic prédictif simplifié")

    if dpe_col is None:
        st.warning("La colonne Etiquette_DPE est nécessaire pour entraîner le modèle prédictif.")
    else:
        candidate_numeric = [
            "Surface_habitable_logement",
            "Hauteur_sous-plafond",
            "Conso_5_usages_é_finale",
            "Emission_GES_5_usages",
            "Ubat_W/m²_K",
            "Besoin_chauffage"
        ]
        candidate_categorical = [
            "Période_construction",
            "Qualité_isolation_enveloppe",
            "Qualité_isolation_murs",
            "Qualité_isolation_menuiseries",
            "Indicateur_confort_été"
        ]

        numeric_features = existing_cols(df_filtered, candidate_numeric)
        categorical_features = existing_cols(df_filtered, candidate_categorical)

        if len(numeric_features) + len(categorical_features) < 2:
            st.warning("Pas assez de variables exploitables pour entraîner le modèle.")
        else:
            model, acc = train_rf_model(
                df_filtered, dpe_col, numeric_features, categorical_features
            )

            if model is None:
                st.warning("Le modèle n’a pas pu être entraîné correctement.")
            else:
                st.metric("Accuracy du modèle", f"{acc:.2f} ({acc*100:.1f} %)")

                st.markdown("### Variables les plus influentes")
                imp = get_feature_importance_table(model)
                if not imp.empty:
                    st.dataframe(imp.head(10), use_container_width=True)

                st.markdown("### Simuler un bâtiment")

                input_data = {}
                cols = st.columns(2)

                i = 0
                for col in numeric_features:
                    with cols[i % 2]:
                        val = float(df_filtered[col].median()) if df_filtered[col].notna().any() else 0.0
                        input_data[col] = st.number_input(col, value=val)
                    i += 1

                for col in categorical_features:
                    with cols[i % 2]:
                        choices = df_filtered[col].dropna().astype(str).unique().tolist()
                        choices = sorted(choices) if choices else ["Non disponible"]
                        input_data[col] = st.selectbox(col, choices)
                    i += 1

                input_df = pd.DataFrame([input_data])

                if st.button("Lancer le diagnostic"):
                    pred = model.predict(input_df)[0]

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(input_df)[0]
                        classes = model.named_steps["rf"].classes_
                        prob_df = pd.DataFrame({
                            "Classe DPE prédite": classes,
                            "Probabilité": probs
                        }).sort_values("Probabilité", ascending=False)
                    else:
                        prob_df = None

                    st.success(f"Classe DPE prédite : **{pred}**")

                    if pred in ["F", "G"]:
                        st.error("Ce profil correspond à une **passoire thermique probable**.")
                    elif pred in ["D", "E"]:
                        st.warning("Ce profil présente une **performance énergétique intermédiaire à fragile**.")
                    else:
                        st.info("Ce profil semble correspondre à un **bâtiment relativement performant**.")

                    if prob_df is not None:
                        fig = px.bar(
                            prob_df,
                            x="Classe DPE prédite",
                            y="Probabilité",
                            title="Probabilités par classe DPE"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        """
Ce module illustre comment une analyse statistique peut être transformée en **outil d’aide au diagnostic**.  
L’objectif n’est pas de remplacer un DPE officiel, mais de proposer une lecture rapide du risque énergétique d’un bâtiment.
"""
                    )