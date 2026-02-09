import streamlit as st
import pandas as pd
import joblib

# 1. Configuration de la page
st.set_page_config(page_title="IA Béton Predictor", layout="centered")

# 2. Chargement des fichiers de production (assure-toi qu'ils sont dans le même dossier)
try:
    model = joblib.load('models_prod/modele_beton_prod.pkl')
    scaler = joblib.load('models_prod/scaler_beton_prod.pkl')
    features_list = joblib.load('models_prod/features_list_prod.pkl')
    st.sidebar.success("Modèle chargé")
except:
    st.error("❌ Fichiers .pkl introuvables. Vérifiez le dossier.")

st.title("Simulateur de Résistance du Béton")
st.markdown("Entrez les dosages pour obtenir une prédiction de la résistance à la compression.")

# --- PRÉPARATION DES DONNÉES ---

# --- 1. COLLECTE DES DONNÉES (TOUT DANS LA SIDEBAR) ---
inputs = {}

st.sidebar.header("Configuration du mélange")

# Dosages
for col in ['Ciment', 'Laitier', 'Cendres', 'Superplastifiant', 'Eau']:
    inputs[col] = st.sidebar.number_input(f"{col} (kg/m³)", min_value=0.0, value=200.0, step=10.0)

st.sidebar.markdown("---")

# Déplacement de l'Âge à gauche
choix_jours = [1, 3, 7, 14, 28, 56, 90]
inputs['Jours'] = st.sidebar.select_slider("Âge du béton (Jours)", options=choix_jours, value=7)

# Variables masquées
inputs['Aggregat_Gros'] = 950.0 
inputs['Aggregat_Fin'] = 750.0

# --- 2. CORPS PRINCIPAL : VISUALISATION DU RATIO ---
st.subheader("Analyse de la formulation")

liant_total = inputs['Ciment'] + inputs['Laitier'] + inputs['Cendres']

if liant_total > 0:
    ratio_el = inputs['Eau'] / liant_total
    
    # Création d'une barre de progression "Slicer" visuelle pour le ratio
    # st.write(f"**Ratio Eau/Liant actuel : {ratio_el:.2f}**")

    # Affichage du ratio avec une taille de police augmentée
    st.markdown(f"""
        <div style="text-align: center; padding: 10px;">
            <p style="font-size: 24px; font-weight: bold; margin-bottom: 0;">
                Ratio Eau/Liant actuel : <span style="color: #1f77b4;">{ratio_el:.2f}</span>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # On affiche une barre colorée selon la zone
    if 0.4 <= ratio_el <= 0.6:
        st.success("Ratio dans les normes de durabilité.")
    else:
        st.warning("Ratio hors zone optimale (Risque de porosité ou de fissures).")
    
    # Astuce visuelle : on utilise une progress bar pour simuler le slicer
    # On limite à 1.0 pour l'affichage
    st.progress(min(ratio_el, 1.0)) 
else:
    st.info("Saisissez des liants à gauche pour calculer le ratio.")

st.markdown("---")

# --- 3. BOUTON DE CALCUL ---
if st.button("Lancer la prédiction de résistance", use_container_width=True):
    df_entree = pd.DataFrame([inputs])[features_list]
    df_entree_scaled = scaler.transform(df_entree)
    prediction = model.predict(df_entree_scaled)[0]
    
    # Affichage stylisé du résultat
    st.balloons() # Petite animation pour le succès
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Résistance Estimée", f"{prediction:.2f} MPa")
    with c2:
        # Traduction en langage simple
        if prediction < 20:
            statut = "Faible"
            couleur = "gray"
        elif prediction < 35:
            statut = "Standard"
            couleur = "green"
        elif prediction < 50:
            statut = "Robuste"
            couleur = "blue"
        else:
            statut = "Ultra-Résistant"
            couleur = "gold"
            
        st.subheader(f"Performance : {statut}")
    

    # --- AJOUT : GRAPHIQUE D'IMPORTANCE ---
    st.markdown("---")
    st.subheader(" Pourquoi ce résultat ?")
    st.write("Voici les facteurs qui ont le plus influencé cette prédiction :")

    # Récupération de l'importance des variables depuis le modèle
    importances = model.feature_importances_
    
    # Création d'un DataFrame pour faciliter le traçage
    feat_imp = pd.DataFrame({
        'Feature': features_list,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    # Affichage du graphique à barres horizontales (natif Streamlit)
    # On affiche les 5 facteurs les plus importants
    st.bar_chart(data=feat_imp.set_index('Feature'), x=None, y='Importance', use_container_width=True)

    st.info("""
    **Interprétation :** En général, le **Ciment** et l'**Âge (Jours)** dominent.
    """)