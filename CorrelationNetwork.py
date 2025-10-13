import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import datetime
import json
import streamlit.components.v1 as components
from concurrent.futures import ThreadPoolExecutor

# NEW: clustering + dendrogram
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

st.set_page_config(page_title="Réseau de corrélation (Ward clustering)", layout="wide")

# Bannière
st.markdown(
    """
    <a href="https://www.zonebourse.com/wbfl/livre" target="_blank">
        <img src="https://raw.githubusercontent.com/EtienneNeptune/AssetAllocation/master/Pubpub.png" width="1500">
    </a>
    """,
    unsafe_allow_html=True
)

# ============================================================
# --- Étape 1 : Chargement des tickers ---
# ============================================================

CAC40_TICKERS = [
    "MC.PA", "RMS.PA", "OR.PA", "AIR.PA", "SU.PA", "SAF.PA", "EL.PA", "TTE.PA", "AI.PA", "SAN.PA",
    "BNP.PA", "CS.PA", "DG.PA", "ACA.PA", "SGO.PA", "HO.PA", "BN.PA", "ENGI.PA", "ORA.PA",
    "DSY.PA", "LR.PA", "KER.PA", "ML.PA", "MT.AS", "VIE.PA", "CAP.PA", "PUB.PA", "STMPA.PA", "EN.PA",
    "BVI.PA", "ERF.PA", "AC.PA", "CA.PA", "RNO.PA", "EDEN.PA", "TEP.PA"
]

@st.cache_data
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    tickers = [r.find_all("td")[0].text.strip() for r in table.find_all("tr")[1:]]
    return tickers

# ============================================================
# --- Titre + Sidebar ---
# ============================================================

st.title("🕸️ Réseau de corrélation — Clustering hiérarchique (Ward)")

with st.sidebar:
    st.header("⚙️ Paramètres")

    index_choice = st.multiselect("Indices :", ["S&P500", "CAC40"], default=["CAC40"])

    tickers = []
    if "CAC40" in index_choice:
        tickers += CAC40_TICKERS
    if "S&P500" in index_choice:
        try:
            tickers += get_sp500_tickers()
        except Exception as e:
            st.error(f"Erreur scraping S&P500 : {e}")

    manual_input = st.text_area("Ajoutez des tickers (séparés par des virgules) :", "")
    if manual_input.strip():
        tickers += [t.strip().upper() for t in manual_input.split(",")]

    tickers = sorted(set(tickers))
    st.write(f"Tickers sélectionnés : **{len(tickers)}**")

    start_date, end_date = st.date_input(
        "Période :", [datetime.date(2024, 1, 1), datetime.date.today()]
    )
    if isinstance(start_date, (list, tuple)):
        start_date, end_date = start_date

    # NEW: choix du nombre de clusters
    n_clusters = st.slider("Nombre de clusters (Ward)", min_value=2, max_value=15, value=6, step=1)

    # Bouton unique: rien ne s'affiche tant qu'on n'a pas cliqué
    launch = st.button("🚀 Lancer le téléchargement et l’analyse")
    reset  = st.button("🗑️ Réinitialiser la session")

if reset:
    for k in ["returns", "metadata", "corr", "linkage", "cluster_labels"]:
        if k in st.session_state:
            del st.session_state[k]
    st.success("Session réinitialisée.")

# ============================================================
# --- Étape 2 : Téléchargement des données ---
# ============================================================

if launch:
    if not tickers:
        st.warning("Veuillez sélectionner au moins un indice ou ajouter des tickers.")
        st.stop()

    # Reset avant nouveau run
    for k in ["returns", "metadata", "corr", "linkage", "cluster_labels"]:
        if k in st.session_state:
            del st.session_state[k]

    st.subheader("📊 Étape 1 : Téléchargement des données journalières")

    with st.spinner("Téléchargement depuis Yahoo Finance..."):
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

    if data.empty:
        st.error("Aucune donnée téléchargée.")
        st.stop()

    # ============================================================
    # --- Étape 3 : Calcul des rendements journaliers ---
    # ============================================================

    try:
        adj_prices = data["Adj Close"].dropna(how="all")
    except Exception:
        # fallback
        adj_prices = data["Close"].dropna(how="all")

    returns = np.log(adj_prices / adj_prices.shift(1)).dropna(how="all")

    # enlever les colonnes entièrement NaN
    returns = returns.dropna(axis=1, how="any")

    if returns.empty or returns.shape[1] < 2:
        st.warning("Pas assez de données pour calculer des corrélations.")
        st.stop()

    @st.cache_data
    def get_metadata(tickers):
        meta = {}
        def fetch(t):
            try:
                info = yf.Ticker(t).info
                return t, info.get("sector", "Inconnu"), info.get("marketCap", np.nan)
            except Exception:
                return t, "Inconnu", np.nan
        with ThreadPoolExecutor(max_workers=10) as executor:
            for t, s, cap in executor.map(fetch, tickers):
                meta[t] = {"sector": s, "marketcap": cap}
        return meta

    metadata = get_metadata(list(returns.columns))

    # Matrice de corrélation
    corr = returns.corr()

    # ✅ Stockage session pour interactivité sans re-télécharger
    st.session_state["returns"] = returns
    st.session_state["metadata"] = metadata
    st.session_state["corr"] = corr

    st.success("✅ Données téléchargées et stockées !")

# ============================================================
# --- Clustering + Réseau + Dendrogramme (réactifs) ---
# ============================================================

def compute_ward_linkage(corr_mat: pd.DataFrame):
    """
    Ward requiert une distance euclidienne. On convertit corr -> distance:
    d_ij = sqrt(2 * (1 - corr_ij)), puis on passe en format condensé.
    """
    C = corr_mat.copy().astype(float)
    C.values[np.diag_indices_from(C)] = 1.0
    C = C.clip(-1.0, 1.0)  # stabilité numérique

    D = np.sqrt(2.0 * (1.0 - C.values))  # matrice NxN
    # convertir en condensed (upper triangle sans diag)
    d_condensed = squareform(D, checks=False)
    Z = hierarchy.linkage(d_condensed, method="ward")
    return Z

def clusters_from_linkage(Z, labels, k):
    cl = hierarchy.fcluster(Z, t=k, criterion="maxclust")
    return dict(zip(labels, cl))  # {ticker: cluster_id}

if "returns" in st.session_state and "corr" in st.session_state:
    returns = st.session_state["returns"]
    metadata = st.session_state["metadata"]
    corr = st.session_state["corr"]

    # --- Clustering Ward ---
    if "linkage" not in st.session_state:
        st.session_state["linkage"] = compute_ward_linkage(corr)

    # recalcul des clusters si n_clusters change
    cluster_labels = clusters_from_linkage(st.session_state["linkage"], list(corr.columns), n_clusters)
    st.session_state["cluster_labels"] = cluster_labels

    st.subheader("🌐 Réseau de corrélation (couleurs = clusters Ward)")

    # Slider de seuil de corrélation pour les liens
    threshold = st.slider("Seuil de corrélation (|ρ| ≥ seuil)", 0.0, 1.0, 0.50, 0.05)

    # Palette couleurs clusters (10 couleurs cyclées)
    # Si >10 clusters, on cyclera; tu peux remplacer par une palette étendue si besoin
    base_colors = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
    ]
    cluster_ids = sorted(set(cluster_labels.values()))
    cluster_color = {cid: base_colors[(cid-1) % len(base_colors)] for cid in cluster_ids}

    # Secteurs = couleur du contour
    sectors = sorted(set(v["sector"] for v in metadata.values()))
    sector_stroke = {s: f"hsl({i * 360 / max(1,len(sectors))}, 60%, 35%)" for i, s in enumerate(sectors)}

    # Taille ~ market cap
    caps = [v["marketcap"] for v in metadata.values() if not np.isnan(v["marketcap"])]
    min_cap, max_cap = (min(caps), max(caps)) if caps else (1, 1)

    nodes, links = [], []
    for t in returns.columns:
        s = metadata[t]["sector"]
        cap = metadata[t]["marketcap"]
        if np.isnan(cap) or max_cap == min_cap:
            radius = 9.0
        else:
            radius = 9.0 + 12.0 * ((cap - min_cap) / (max_cap - min_cap))
        c_id = cluster_labels.get(t, 1)
        fill_col = cluster_color.get(c_id, "#999")
        stroke_col = sector_stroke.get(s, "#444")
        nodes.append({
            "id": t,
            # 👇 étiquette visible sur le graphe : Ticker + Cluster + Secteur
            "name": f"{t} (C{c_id}) · {s}",
            "radius": radius,
            "color": fill_col,
            "stroke": stroke_col,
            # 👇 utile pour le tooltip
            "sector": s,
            "cluster": int(c_id),
        })

    for i in range(len(returns.columns)):
        for j in range(i + 1, len(returns.columns)):
            a, b = returns.columns[i], returns.columns[j]
            c = corr.loc[a, b]
            if not pd.isna(c) and abs(c) >= threshold:
                # value -> plus grand si corr plus faible => lien plus long
                value = float(5 * (1 - (c + 1) / 2))
                links.append({"source": a, "target": b, "value": value})

    graph_payload = {"nodes": nodes, "links": links}

    # ------ Graphe D3 (zoom/pan, contours = secteurs) ------
    html = f"""
    <div id="graph" style="width:100%; height:650px; border:1px solid #ddd; border-radius:8px;"></div>
    <style>
      .node-label {{
        font: 11px ui-sans-serif, system-ui;
        pointer-events: none;
        /* la couleur sera forcée via JS selon le thème */
      }}
      .tooltip {{
        position: absolute;
        padding: 6px 10px;
        background: rgba(255,255,255,0.95);
        border: 1px solid #ccc;
        border-radius: 6px;
        font-size: 12px;
        pointer-events: none;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
      }}
    </style>

    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <script>
      const DATA = {json.dumps(graph_payload).replace("</", "<\\/")};
      const container = document.getElementById('graph');
      const WIDTH = container.clientWidth || 1000;
      const HEIGHT = container.clientHeight || 650;

      const outer = d3.select(container).append("svg")
        .attr("width", WIDTH)
        .attr("height", HEIGHT);

      const g = outer.append("g");
      outer.call(d3.zoom().scaleExtent([0.3, 3]).on("zoom", (e) => g.attr("transform", e.transform)));

      // Elements
      const tooltip = d3.select(container).append("div").attr("class", "tooltip").style("opacity", 0);

      const link = g.append("g")
        .attr("stroke-opacity", 0.6)
        .selectAll("line")
        .data(DATA.links)
        .join("line")
        .attr("stroke-width", d => Math.sqrt(1 + d.value));

      const node = g.append("g")
        .attr("stroke-width", 2.0)
        .selectAll("circle")
        .data(DATA.nodes)
        .join("circle")
        .attr("r", d => d.radius || 8)
        .attr("fill", d => d.color)
        .attr("stroke", d => d.stroke || "#444")
        .style("cursor", "grab")
        .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended))
        .on("mouseover", (event, d) => {{
          tooltip.transition().duration(80).style("opacity", 1);
          const name = d.name || d.id;
          tooltip.html(`<b>${{name}}</b>`)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 18) + "px");
        }})
        .on("mousemove", (event) => {{
          tooltip.style("left", (event.pageX + 10) + "px").style("top", (event.pageY - 18) + "px");
        }})
        .on("mouseout", () => tooltip.transition().duration(120).style("opacity", 0));

      const labels = g.append("g")
        .selectAll("text")
        .data(DATA.nodes)
        .join("text")
        .attr("class", "node-label")
        .text(d => d.name || d.id)
        .attr("text-anchor", "middle")
        .attr("dy", -10);

      const simulation = d3.forceSimulation(DATA.nodes)
        .force("link", d3.forceLink(DATA.links).id(d => d.id).distance(d => 100 + 50*d.value).strength(0.6))
        .force("charge", d3.forceManyBody().strength(-220).distanceMax(400))
        .force("center", d3.forceCenter(WIDTH/2, HEIGHT/2))
        .force("collision", d3.forceCollide().radius(d => (d.radius || 8) + 6))
        .on("tick", ticked);

      function ticked() {{
        link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        node.attr("cx", d => d.x).attr("cy", d => d.y);
        labels.attr("x", d => d.x).attr("y", d => d.y - 10);
      }}

      function dragstarted(event, d) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
        d3.select(this).style("cursor", "grabbing");
      }}
      function dragged(event, d) {{ d.fx = event.x; d.fy = event.y; }}
      function dragended(event, d) {{
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
        d3.select(this).style("cursor", "grab");
      }}

      // --------- THEME AWARE (clair/sombre) ----------
      const mq = window.matchMedia('(prefers-color-scheme: dark)');
      function applyTheme(isDark) {{
        // Couleur des labels (texte)
        labels.attr("fill", isDark ? "#e5e7eb" : "#333"); // gris clair en dark, gris foncé en clair
        // Couleur des liens
        link.attr("stroke", isDark ? "#6b7280" : "#999"); // gris-500 vs gris-600
        // Contour du conteneur
        container.style.borderColor = isDark ? "#374151" : "#ddd";
        // Tooltip
        const ttBG = isDark ? "rgba(31,41,55,0.95)" : "rgba(255,255,255,0.95)"; // slate-800 vs blanc
        const ttFG = isDark ? "#e5e7eb" : "#111827"; // texte
        const ttBD = isDark ? "#4b5563" : "#d1d5db";
        d3.select(container).select(".tooltip")
          .style("background", ttBG)
          .style("color", ttFG)
          .style("border-color", ttBD)
          .style("box-shadow", isDark ? "0 2px 6px rgba(0,0,0,0.5)" : "0 2px 6px rgba(0,0,0,0.08)");
      }}
      // Appliquer au chargement
      applyTheme(mq.matches);
      // Réagir quand l’utilisateur change de thème
      mq.addEventListener ? mq.addEventListener('change', e => applyTheme(e.matches))
                          : mq.addListener(e => applyTheme(e.matches)); // fallback anciens navigateurs
    </script>
    """
    components.html(html, height=700, scrolling=True)

    # ---------------- Dendrogramme hiérarchique (Ward) ----------------
    st.subheader("🌳 Dendrogramme hiérarchique (Ward)")

    # Recalcul (ou reuse) de la linkage si nécessaire (déjà calculée plus haut)
    Z = st.session_state["linkage"]

    fig, ax = plt.subplots(figsize=(12, 5))
    # labels = tickers du corr
    labels = list(corr.columns)
    hierarchy.dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_ylabel("Distance (Ward)")
    ax.set_title("Dendrogramme complet (ascendant hiérarchique, Ward)")
    st.pyplot(fig)

else:
    st.info("👉 Configurez vos paramètres dans la barre latérale puis cliquez sur **Lancer le téléchargement et l’analyse** pour construire le réseau + dendrogramme.")
