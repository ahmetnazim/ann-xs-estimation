/**
 * ANN-XS Estimation WebGIS — Single-Page Interactive Client Application
 *
 * Renders:
 *  1. Interactive CONUS & USACE eHydro WebGIS Map (MapLibre GL JS + SVG fallback)
 *  2. Analytical Dingman Cross-Section (XS) Profile z(x) = -d_max(1 - |2x/W|^r) + 90% UQ + 3DEP Lidar Inset
 *  3. Live OPS-1 Stage–Discharge Rating Curve Simulator (Constant-n vs. Dynamic-n LODO gamma)
 *  4. Confirmatory Geometry Benchmark (Event 1–3, Tables T1–T9, H1–H4, Ablations)
 *  5. Operational Stage Routing & CCHE1D 1D Solver Readiness (OPS-1 v1–v3, Tables T10–T11)
 *  6. Data Provenance, 14-Rule QA/QC Cascade & 7-Event Cryptographic Ledger
 */

const MODEL_COLORS = {
  "M2-XGB": "#0072B2",
  "M2-MLP": "#56B4E9",
  "M5": "#D55E00",
  "M4": "#E69F00",
  "M1": "#009E73",
  "M3": "#CC79A7",
  "M0": "#7F7F7F",
  "M-null": "#94A3B8",
  "M6-NN": "#8B5CF6",
  "M6-XGB": "#EC4899",
  "A_J": "#14B8A6",
  "Oracle": "#111827",
  "CCHE1D-Hybrid": "#0F7C80",
};

const HEADLINE_MODELS = ["M2-XGB", "M2-MLP", "M1", "M5", "M4", "M3", "M0"];
const ALL_TEN_MODELS = ["M-null", "M0", "M1", "M2-MLP", "M2-XGB", "M3", "M4", "M5", "M6-NN", "M6-XGB"];

const BASEMAP_STYLES = {
  positron: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  dark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  topo: {
    version: 8,
    sources: {
      usgs: {
        type: "raster",
        tiles: ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"],
        tileSize: 256,
        attribution: "USGS The National Map",
      },
    },
    layers: [{ id: "usgs-topo", type: "raster", source: "usgs" }],
  },
};

const state = {
  benchmarks: null,
  physio: null,
  conus: null,
  ehydro: null,
  siteById: new Map(),
  ehydroById: new Map(),
  selectedSite: null,
  selectedIsEhydro: false,
  xsSplit: "sid",
  activeModels: new Set(["M2-XGB", "M2-MLP", "M5", "M1"]),
  showUqBand: true,
  show3depInset: true,
  simQRatio: 3.5,
  simRoughnessMode: "dyn",
  map: null,
  colorMode: "win_geom",
  singleModel: "M2-XGB",
  filterDivision: "ALL",
  filterSlope: "ALL",
  filterAvailability: "ALL",
  decisiveOnly: true,
  showPhysio: true,
  showEhydro: true,
};

function fmt(val, digits = 3, fallback = "—") {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return fallback;
  return Number(val).toFixed(digits);
}

function svgEl(tag, attrs = {}, text = null) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  }
  if (text !== null) el.textContent = text;
  return el;
}

/* ==========================================================================
   1. Initialization & Data Loading
   ========================================================================== */

async function initApp() {
  setupThemeToggle();
  setupDialogFallbacks();
  setupNavigation();

  const [benchmarks, physio, conus, ehydro] = await Promise.all([
    fetch("data/benchmarks.json").then((r) => r.json()),
    fetch("data/physio_divisions.geojson").then((r) => r.json()),
    fetch("data/sites_conus.json").then((r) => r.json()),
    fetch("data/sites_ehydro.json").then((r) => r.json()),
  ]);

  state.benchmarks = benchmarks;
  state.physio = physio;
  state.conus = conus;
  state.ehydro = ehydro;

  for (const s of conus.sites) state.siteById.set(s.id, s);
  for (const loc of ehydro.locations) state.ehydroById.set(loc.id, loc);

  populateFiltersAndSearch();
  setupShowcasePresets();
  setupXsControls();
  setupMapControls();
  initMapView();

  // Select default showcase reach (normal-slope mainstem with 3DEP + OPS-1)
  const defaultSite =
    conus.sites.find((s) => s.fp && s.fs === 0 && s.nh >= 8 && s.da > 800 && s.da < 15000) ||
    conus.sites[0];
  selectReach(defaultSite.id, false, false);

  // Populate dashboard views
  initGeometryBenchmarkView();
  initOperationalView();
  initProvenanceView();
  populateLedgerDialog();
}

/* ==========================================================================
   2. Theme Toggle & Dialog Fallbacks (modern-web-guidance compliant)
   ========================================================================== */

function setupThemeToggle() {
  const btn = document.getElementById("theme-toggle-btn");
  const meta = document.querySelector('meta[name="color-scheme"]');
  if (!btn || !meta) return;

  btn.addEventListener("click", () => {
    const sysDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    let stored = null;
    try {
      stored = localStorage.getItem("color-scheme");
    } catch (_) {}
    const currentRenderedDark = stored ? stored === "dark" : sysDark;
    const target = currentRenderedDark ? "light" : "dark";
    const sysPref = sysDark ? "dark" : "light";

    try {
      if (target === sysPref) {
        localStorage.removeItem("color-scheme");
        meta.content = "light dark";
      } else {
        localStorage.setItem("color-scheme", target);
        meta.content = target;
      }
    } catch (_) {
      meta.content = target === sysPref ? "light dark" : target;
    }
  });

  window.addEventListener("storage", (e) => {
    if (e.key === "color-scheme") {
      meta.content = e.newValue ?? "light dark";
    }
  });
}

function setupDialogFallbacks() {
  const ledgerBtn = document.getElementById("open-ledger-btn");
  const ledgerDialog = document.getElementById("ledger-dialog");
  if (ledgerBtn && ledgerDialog) {
    ledgerBtn.addEventListener("click", () => ledgerDialog.showModal());
    ledgerDialog.addEventListener("close", () => {
      ledgerDialog.returnValue = "";
    });
  }

  // Fallback for browsers where <dialog closedby="any"> is not yet supported
  if (typeof HTMLDialogElement !== "undefined" && !("closedBy" in HTMLDialogElement.prototype)) {
    const outside = (e, dialog) => {
      const r = dialog.getBoundingClientRect();
      return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
    };
    for (const dialog of document.querySelectorAll('dialog[closedby="any"]')) {
      let pressedOutside = false;
      dialog.addEventListener("pointerdown", (e) => {
        pressedOutside = outside(e, dialog);
      });
      dialog.addEventListener("pointerup", (e) => {
        if (pressedOutside && outside(e, dialog)) dialog.close();
      });
    }
  }
}

function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  const views = {
    map: document.getElementById("view-map"),
    geometry: document.getElementById("view-geometry"),
    operational: document.getElementById("view-operational"),
    provenance: document.getElementById("view-provenance"),
  };

  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      const target = tab.dataset.view;
      for (const t of tabs) {
        const active = t === tab;
        t.classList.toggle("is-active", active);
        t.setAttribute("aria-pressed", active ? "true" : "false");
      }
      for (const [k, el] of Object.entries(views)) {
        if (!el) continue;
        if (k === target) {
          el.removeAttribute("hidden");
          el.classList.add("is-active");
        } else {
          el.setAttribute("hidden", "");
          el.classList.remove("is-active");
        }
      }
      if (target === "map" && state.map) {
        setTimeout(() => state.map.resize(), 60);
      }
    });
  }
}

/* ==========================================================================
   3. Search, Filters & Diagnostic Showcase Presets
   ========================================================================== */

function populateFiltersAndSearch() {
  const divSelect = document.getElementById("filter-division");
  const divisions = Array.from(new Set(state.conus.sites.map((s) => s.d))).sort();
  for (const d of divisions) {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    divSelect.appendChild(opt);
  }

  const datalist = document.getElementById("site-suggestions");
  const sampleSites = state.conus.sites.slice(0, 250);
  for (const s of sampleSites) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.label = `USGS ${s.id} · COMID ${s.c} · ${s.d} (DA ${s.da} km²)`;
    datalist.appendChild(opt);
  }
  for (const loc of state.ehydro.locations) {
    const opt = document.createElement("option");
    opt.value = loc.id;
    opt.label = `USACE eHydro ${loc.id} · ${loc.river} (COMID ${loc.comid})`;
    datalist.appendChild(opt);
  }

  const form = document.getElementById("site-search-form");
  const input = document.getElementById("site-search-input");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    if (state.siteById.has(q)) {
      selectReach(q, false, true);
      return;
    }
    if (state.ehydroById.has(q)) {
      selectReach(q, true, true);
      return;
    }
    const byComid = state.conus.sites.find((s) => String(s.c) === q);
    if (byComid) {
      selectReach(byComid.id, false, true);
      return;
    }
    const byRiver = state.ehydro.locations.find(
      (l) => l.river.toLowerCase().includes(q.toLowerCase()) || l.id.toLowerCase().includes(q.toLowerCase())
    );
    if (byRiver) {
      selectReach(byRiver.id, true, true);
    }
  });
}

function setupShowcasePresets() {
  const container = document.getElementById("showcase-buttons");
  const sNormal =
    state.conus.sites.find((s) => s.fp && s.fs === 0 && s.w[2][0] === "M5" && s.nh >= 5) ||
    state.conus.sites[10];
  const sFloored =
    state.conus.sites.find((s) => s.fp && s.fs === 1 && s.w[0][0] === "M2-XGB") ||
    state.conus.sites.find((s) => s.fs === 1);
  const sHeadwater =
    state.conus.sites.find((s) => s.da < 35 && s.fp && s.fs === 0) || state.conus.sites[0];
  const sEhydro =
    state.ehydro.locations.find((l) => l.best_area_model === "M5" && l.single_comid_uq) ||
    state.ehydro.locations[0];

  const presets = [
    { label: "M5 Normal-Slope Winner", id: sNormal.id, isEh: false },
    { label: "Floored Slope (S = 10⁻⁵)", id: sFloored.id, isEh: false },
    { label: "Headwater 3DEP Reach", id: sHeadwater.id, isEh: false },
    { label: `USACE eHydro (${sEhydro.river})`, id: sEhydro.id, isEh: true },
  ];

  for (const p of presets) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "preset-btn";
    btn.textContent = p.label;
    btn.addEventListener("click", () => {
      document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      selectReach(p.id, p.isEh, true);
    });
    container.appendChild(btn);
  }
}

/* ==========================================================================
   4. Interactive WebGIS Map (MapLibre GL JS + Color/Filter Engine)
   ========================================================================== */

function siteMatchesFilters(s) {
  if (state.filterDivision !== "ALL" && s.d !== state.filterDivision) return false;
  if (state.filterSlope === "NORMAL" && s.fs !== 0) return false;
  if (state.filterSlope === "FLOORED" && s.fs !== 1) return false;
  if (state.filterAvailability === "3DEP" && !s.fp) return false;
  if (state.filterAvailability === "OPS1" && s.no === 0) return false;
  if (state.filterAvailability === "TAIL_DA" && s.tda !== 1) return false;
  if (state.filterAvailability === "TAIL_SO" && s.tso !== 1) return false;
  return true;
}

function computeSiteColor(s) {
  const mode = state.colorMode;
  if (mode === "win_geom" || mode === "win_op_fp" || mode === "win_op_dyn") {
    const idx = mode === "win_geom" ? 0 : mode === "win_op_fp" ? 1 : 2;
    const winArr = s.w[idx];
    if (!winArr || !winArr[0]) return "#CBD5E1";
    const [best, , , , decisive] = winArr;
    if (state.decisiveOnly && !decisive) return "#94A3B8";
    return MODEL_COLORS[best] || "#64748B";
  }
  if (mode === "slope_regime") {
    return s.fs === 1 ? "#D55E00" : "#0F7C80";
  }
  if (mode === "data_tier") {
    if (s.fp) return "#0F7C80";
    if (s.no > 0) return "#0072B2";
    return "#94A3B8";
  }
  if (mode === "latent_n") {
    const m5 = s.p.sid[HEADLINE_MODELS.indexOf("M5")];
    const nHat = m5 ? m5[5] : 0.035;
    if (nHat < 0.025) return "#38BDF8";
    if (nHat < 0.045) return "#0F7C80";
    if (nHat < 0.075) return "#E69F00";
    return "#D55E00";
  }
  if (mode === "latent_r") {
    const m5 = s.p.sid[HEADLINE_MODELS.indexOf("M5")];
    const rHat = m5 ? m5[4] : 1.8;
    if (rHat < 1.2) return "#8B5CF6";
    if (rHat < 1.8) return "#0072B2";
    if (rHat < 2.5) return "#009E73";
    return "#E69F00";
  }
  if (mode === "model_rmse") {
    const mIdx = ALL_TEN_MODELS.indexOf(state.singleModel);
    const rmArr = s.rm[0];
    const val = rmArr && mIdx >= 0 ? rmArr[mIdx] : null;
    if (val === null) return "#94A3B8";
    if (val < 0.25) return "#1B7F3B";
    if (val < 0.45) return "#0F7C80";
    if (val < 0.70) return "#E69F00";
    return "#B22222";
  }
  return "#0072B2";
}

function buildSitesGeoJSON() {
  const features = [];
  for (const s of state.conus.sites) {
    if (!siteMatchesFilters(s)) continue;
    const winGeom = s.w[0] ? s.w[0][0] : "—";
    const winDyn = s.w[2] ? s.w[2][0] : "—";
    features.push({
      type: "Feature",
      geometry: { type: "Point", coordinates: [s.lo, s.la] },
      properties: {
        id: s.id,
        comid: s.c,
        div: s.d,
        da: s.da,
        color: computeSiteColor(s),
        radius: Math.max(3.5, Math.min(8.0, 2.6 + Math.log10(Math.max(s.da, 1)) * 0.95)),
        win_geom: winGeom,
        win_dyn: winDyn,
      },
    });
  }
  const badge = document.getElementById("visible-count-badge");
  if (badge) {
    badge.textContent = `${features.length.toLocaleString()} gauges · ${
      state.showEhydro ? state.ehydro.n_locations : 0
    } eHydro`;
  }
  return { type: "FeatureCollection", features };
}

function buildEhydroGeoJSON() {
  return {
    type: "FeatureCollection",
    features: state.ehydro.locations.map((loc) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [loc.lon, loc.lat] },
      properties: {
        id: loc.id,
        river: loc.river,
        comid: loc.comid,
        color: MODEL_COLORS[loc.best_area_model] || "#D55E00",
      },
    })),
  };
}

function updateMapLegend() {
  const legend = document.getElementById("map-legend");
  if (!legend) return;
  const mode = state.colorMode;
  let title = "";
  let items = [];

  if (mode.startsWith("win_")) {
    const labelMap = {
      win_geom: "Reach Winner: Anchor Geometry (S-ID)",
      win_op_dyn: "Reach Winner: Flood Stage Dynamic n (OPS-1 v3)",
      win_op_fp: "Reach Winner: Flood Stage 3DEP Const. n (OPS-1 v2)",
    };
    title = labelMap[mode];
    items = ["M2-XGB", "M2-MLP", "M1", "M5", "M4", "M3", "M0", "M-null"].map((m) => ({
      label: m,
      color: MODEL_COLORS[m],
    }));
    if (state.decisiveOnly) {
      items.push({ label: "Tie (≤ margin)", color: "#94A3B8" });
    }
  } else if (mode === "slope_regime") {
    title = "NHDPlusV2.1 Slope-Floor Regime (F_S)";
    items = [
      { label: "Normal S > 10⁻⁵ (79.1% · M4/M5 win)", color: "#0F7C80" },
      { label: "Floored S = 10⁻⁵ (20.9% · Clamped)", color: "#D55E00" },
    ];
  } else if (mode === "data_tier") {
    title = "Reach Data Availability Tier";
    items = [
      { label: "3DEP Lidar + Stage (n = 3,282)", color: "#0F7C80" },
      { label: "OPS-1 Multi-Stage (n = 3,643)", color: "#0072B2" },
      { label: "Anchor ADCP Only (n = 345)", color: "#94A3B8" },
    ];
  } else if (mode === "latent_n") {
    title = "M5 Predicted Manning Roughness n̂";
    items = [
      { label: "< 0.025 (Smooth)", color: "#38BDF8" },
      { label: "0.025 – 0.045", color: "#0F7C80" },
      { label: "0.045 – 0.075", color: "#E69F00" },
      { label: "> 0.075 (Rough / Headwater)", color: "#D55E00" },
    ];
  } else if (mode === "latent_r") {
    title = "M5 Predicted Dingman Shape r̂";
    items = [
      { label: "r̂ < 1.2 (V-Notch)", color: "#8B5CF6" },
      { label: "1.2 – 1.8", color: "#0072B2" },
      { label: "1.8 – 2.5 (Parabolic)", color: "#009E73" },
      { label: "r̂ > 2.5 (Wide/Flat)", color: "#E69F00" },
    ];
  } else if (mode === "model_rmse") {
    title = `${state.singleModel} Anchor Geometry RMSE_log`;
    items = [
      { label: "< 0.25 (Excellent)", color: "#1B7F3B" },
      { label: "0.25 – 0.45", color: "#0F7C80" },
      { label: "0.45 – 0.70", color: "#E69F00" },
      { label: "> 0.70 (High Error)", color: "#B22222" },
    ];
  }

  if (state.showEhydro) {
    items.push({ label: "USACE eHydro S-IND (◆)", color: "#D55E00", triangle: true });
  }

  legend.innerHTML = `
    <div class="legend-title">${title}</div>
    <div class="legend-grid">
      ${items
        .map(
          (it) => `
        <div class="legend-item">
          <span class="legend-swatch ${it.triangle ? "is-triangle" : ""}" style="background-color:${it.color}"></span>
          <span>${it.label}</span>
        </div>`
        )
        .join("")}
    </div>
  `;
}

function refreshMapSources() {
  updateMapLegend();
  if (!state.map || !state.map.isStyleLoaded()) return;
  const conusSrc = state.map.getSource("conus-sites");
  if (conusSrc) conusSrc.setData(buildSitesGeoJSON());

  if (state.map.getLayer("physio-fill")) {
    const vis = state.showPhysio ? "visible" : "none";
    state.map.setLayoutProperty("physio-fill", "visibility", vis);
    state.map.setLayoutProperty("physio-line", "visibility", vis);
  }
  if (state.map.getLayer("ehydro-points")) {
    state.map.setLayoutProperty("ehydro-points", "visibility", state.showEhydro ? "visible" : "none");
  }
}

function addDataLayersToMap(map) {
  if (!map.getSource("physio-divisions")) {
    map.addSource("physio-divisions", { type: "geojson", data: state.physio });
  }
  if (!map.getLayer("physio-fill")) {
    map.addLayer({
      id: "physio-fill",
      type: "fill",
      source: "physio-divisions",
      paint: {
        "fill-color": "#0072B2",
        "fill-opacity": 0.08,
      },
    });
    map.addLayer({
      id: "physio-line",
      type: "line",
      source: "physio-divisions",
      paint: {
        "line-color": "#475569",
        "line-width": 1.2,
        "line-dasharray": [3, 2],
      },
    });
  }

  if (!map.getSource("conus-sites")) {
    map.addSource("conus-sites", { type: "geojson", data: buildSitesGeoJSON() });
  }
  if (!map.getLayer("conus-points")) {
    map.addLayer({
      id: "conus-points",
      type: "circle",
      source: "conus-sites",
      paint: {
        "circle-radius": ["get", "radius"],
        "circle-color": ["get", "color"],
        "circle-stroke-color": "#0f172a",
        "circle-stroke-width": 0.65,
        "circle-opacity": 0.88,
      },
    });
  }

  if (!map.getSource("ehydro-sites")) {
    map.addSource("ehydro-sites", { type: "geojson", data: buildEhydroGeoJSON() });
  }
  if (!map.getLayer("ehydro-points")) {
    map.addLayer({
      id: "ehydro-points",
      type: "circle",
      source: "ehydro-sites",
      paint: {
        "circle-radius": 6.5,
        "circle-color": ["get", "color"],
        "circle-stroke-color": "#ffffff",
        "circle-stroke-width": 1.8,
      },
    });
  }

  if (!map.getSource("selected-point")) {
    map.addSource("selected-point", {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] },
    });
  }
  if (!map.getLayer("selected-ring")) {
    map.addLayer({
      id: "selected-ring",
      type: "circle",
      source: "selected-point",
      paint: {
        "circle-radius": 11,
        "circle-color": "transparent",
        "circle-stroke-color": "#D55E00",
        "circle-stroke-width": 3,
      },
    });
  }
}

function initMapView() {
  updateMapLegend();
  const container = document.getElementById("map-canvas");
  if (!container || typeof maplibregl === "undefined") return;

  try {
    const map = new maplibregl.Map({
      container: "map-canvas",
      style: BASEMAP_STYLES.positron,
      center: [-96.5, 38.5],
      zoom: 3.6,
      attributionControl: true,
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-left");
    state.map = map;

    map.on("load", () => {
      addDataLayersToMap(map);
      updateSelectedMapHighlight();
    });

    const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false });

    map.on("mouseenter", "conus-points", (e) => {
      map.getCanvas().style.cursor = "pointer";
      const f = e.features[0];
      const p = f.properties;
      popup
        .setLngLat(f.geometry.coordinates)
        .setHTML(
          `<strong>USGS ${p.id}</strong> (COMID ${p.comid})<br>` +
            `${p.div} · DA: ${Number(p.da).toLocaleString()} km²<br>` +
            `Geom Best: <b>${p.win_geom}</b> · Dyn-n Best: <b>${p.win_dyn}</b>`
        )
        .addTo(map);
    });
    map.on("mouseleave", "conus-points", () => {
      map.getCanvas().style.cursor = "";
      popup.remove();
    });
    map.on("click", "conus-points", (e) => {
      const f = e.features[0];
      if (f && f.properties && f.properties.id) {
        selectReach(String(f.properties.id), false, false);
      }
    });

    map.on("mouseenter", "ehydro-points", (e) => {
      map.getCanvas().style.cursor = "pointer";
      const f = e.features[0];
      const p = f.properties;
      popup
        .setLngLat(f.geometry.coordinates)
        .setHTML(`<strong>USACE eHydro ${p.id}</strong><br>River: ${p.river} (COMID ${p.comid})`)
        .addTo(map);
    });
    map.on("mouseleave", "ehydro-points", () => {
      map.getCanvas().style.cursor = "";
      popup.remove();
    });
    map.on("click", "ehydro-points", (e) => {
      const f = e.features[0];
      if (f && f.properties && f.properties.id) {
        selectReach(String(f.properties.id), true, false);
      }
    });
  } catch (_) {
    // Graceful fallback if WebGL is disabled in headless test runners
  }
}

function setupMapControls() {
  const colorSel = document.getElementById("map-color-mode");
  const singleModCtrl = document.getElementById("single-model-control");
  const singleModSel = document.getElementById("map-single-model");
  const divSel = document.getElementById("filter-division");
  const slopeSel = document.getElementById("filter-slope");
  const availSel = document.getElementById("filter-availability");
  const baseSel = document.getElementById("map-basemap");
  const decChk = document.getElementById("toggle-decisive-only");
  const physChk = document.getElementById("toggle-physio-layer");
  const ehChk = document.getElementById("toggle-ehydro-layer");

  colorSel.addEventListener("change", () => {
    state.colorMode = colorSel.value;
    singleModCtrl.classList.toggle("is-hidden", state.colorMode !== "model_rmse");
    refreshMapSources();
  });
  singleModSel.addEventListener("change", () => {
    state.singleModel = singleModSel.value;
    refreshMapSources();
  });
  divSel.addEventListener("change", () => {
    state.filterDivision = divSel.value;
    refreshMapSources();
  });
  slopeSel.addEventListener("change", () => {
    state.filterSlope = slopeSel.value;
    refreshMapSources();
  });
  availSel.addEventListener("change", () => {
    state.filterAvailability = availSel.value;
    refreshMapSources();
  });
  decChk.addEventListener("change", () => {
    state.decisiveOnly = decChk.checked;
    refreshMapSources();
  });
  physChk.addEventListener("change", () => {
    state.showPhysio = physChk.checked;
    refreshMapSources();
  });
  ehChk.addEventListener("change", () => {
    state.showEhydro = ehChk.checked;
    refreshMapSources();
  });
  baseSel.addEventListener("change", () => {
    if (!state.map) return;
    const style = BASEMAP_STYLES[baseSel.value] || BASEMAP_STYLES.positron;
    state.map.setStyle(style);
    state.map.once("styledata", () => {
      addDataLayersToMap(state.map);
      updateSelectedMapHighlight();
    });
  });
}

function updateSelectedMapHighlight() {
  if (!state.map || !state.map.isStyleLoaded() || !state.selectedSite) return;
  const src = state.map.getSource("selected-point");
  if (!src) return;
  const lon = state.selectedIsEhydro ? state.selectedSite.lon : state.selectedSite.lo;
  const lat = state.selectedIsEhydro ? state.selectedSite.lat : state.selectedSite.la;
  src.setData({
    type: "FeatureCollection",
    features: [{ type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties: {} }],
  });
}

/* ==========================================================================
   5. Predicted Cross-Section (XS) & Stage-Discharge Routing Inspector
   ========================================================================== */

function setupXsControls() {
  const chipBar = document.getElementById("xs-model-toggles");
  for (const m of HEADLINE_MODELS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `model-chip ${state.activeModels.has(m) ? "is-active" : ""}`;
    btn.style.setProperty("--chip-color", MODEL_COLORS[m]);
    btn.innerHTML = `<span class="chip-dot"></span><span>${m}</span>`;
    btn.addEventListener("click", () => {
      if (state.activeModels.has(m)) {
        if (state.activeModels.size > 1) state.activeModels.delete(m);
      } else {
        state.activeModels.add(m);
      }
      btn.classList.toggle("is-active", state.activeModels.has(m));
      renderXsInspector();
    });
    chipBar.appendChild(btn);
  }

  const splitSel = document.getElementById("xs-split-select");
  splitSel.addEventListener("change", () => {
    state.xsSplit = splitSel.value;
    renderXsInspector();
  });

  const uqChk = document.getElementById("toggle-uq-band");
  uqChk.addEventListener("change", () => {
    state.showUqBand = uqChk.checked;
    renderXsPlot();
  });

  const fpChk = document.getElementById("toggle-3dep-inset");
  fpChk.addEventListener("change", () => {
    state.show3depInset = fpChk.checked;
    renderXsPlot();
  });

  const qSlider = document.getElementById("sim-q-slider");
  const rMode = document.getElementById("sim-roughness-mode");
  qSlider.addEventListener("input", () => {
    state.simQRatio = Number(qSlider.value);
    renderRatingSimulator();
  });
  rMode.addEventListener("change", () => {
    state.simRoughnessMode = rMode.value;
    renderRatingSimulator();
  });

  const exportBtn = document.getElementById("export-site-btn");
  exportBtn.addEventListener("click", () => {
    if (!state.selectedSite) return;
    const blob = new Blob([JSON.stringify(state.selectedSite, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `annxs_${state.selectedSite.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

function selectReach(id, isEhydro = false, fly = false) {
  const site = isEhydro ? state.ehydroById.get(id) : state.siteById.get(id);
  if (!site) return;
  state.selectedSite = site;
  state.selectedIsEhydro = isEhydro;
  updateSelectedMapHighlight();

  if (fly && state.map) {
    const lon = isEhydro ? site.lon : site.lo;
    const lat = isEhydro ? site.lat : site.la;
    state.map.flyTo({ center: [lon, lat], zoom: 6.5, duration: 900 });
  }
  renderXsInspector();
}

function getSiteModelEntry(site, isEhydro, modelName) {
  if (isEhydro) {
    return site.pred ? site.pred[modelName] : null;
  }
  const idx = HEADLINE_MODELS.indexOf(modelName);
  if (idx < 0) return null;
  const pack = site.p[state.xsSplit] || site.p.sid;
  return pack ? pack[idx] : null;
}

function renderXsInspector() {
  const s = state.selectedSite;
  if (!s) return;
  const isEh = state.selectedIsEhydro;

  const badgesEl = document.getElementById("xs-site-badges");
  const titleEl = document.getElementById("xs-site-title");
  const subEl = document.getElementById("xs-site-subtitle");

  if (isEh) {
    titleEl.textContent = `USACE eHydro ${s.id} — ${s.river}`;
    subEl.textContent = `Independent Survey (S-IND) · Outlet COMID ${s.comid} · ${s.n_comids} reach(es) · Length ${s.length_km} km`;
    badgesEl.innerHTML = `
      <span class="badge-pill is-good">S-IND USACE eHydro</span>
      <span class="badge-pill">Best Area Model: ${s.best_area_model || "M5"}</span>
      ${s.single_comid_uq ? '<span class="badge-pill is-good">Single-COMID UQ Cohort</span>' : ""}
    `;
  } else {
    titleEl.textContent = `USGS Gauge ${s.id} (COMID ${s.c})`;
    subEl.textContent = `${s.d} · Stream Order ${s.so} · Drainage Area ${Number(s.da).toLocaleString()} km² · (${s.la}°, ${s.lo}°)`;
    const slopeBadge =
      s.fs === 1
        ? `<span class="badge-pill is-warn" title="NHDPlusV2.1 slope clamped at S = 10⁻⁵">Floored Slope S = 10⁻⁵ (F_S=1)</span>`
        : `<span class="badge-pill is-good">Normal Slope S = ${s.s}</span>`;
    const fpBadge = s.fp
      ? `<span class="badge-pill is-good">3DEP Lidar Transect (${s.fp[0]} m)</span>`
      : `<span class="badge-pill">In-Channel Only</span>`;
    const winGeom = s.w[0] && s.w[0][0] ? `Geom Winner: ${s.w[0][0]}` : "Geom: Tie";
    badgesEl.innerHTML = `${slopeBadge}${fpBadge}<span class="badge-pill">${winGeom}</span>`;
  }

  // Telemetry boxes
  const covEl = document.getElementById("site-covariate-cards");
  const [wObs, dbarObs, dmaxObs, aObs, vObs, rObs] = s.obs;
  covEl.innerHTML = `
    <div class="telemetry-box"><span>Obs Width W</span><strong>${fmt(wObs, 1)} m</strong></div>
    <div class="telemetry-box"><span>Obs Mean d̄</span><strong>${fmt(dbarObs, 2)} m</strong></div>
    <div class="telemetry-box"><span>Obs Max d_max</span><strong>${fmt(dmaxObs, 2)} m</strong></div>
    <div class="telemetry-box"><span>Obs Area A</span><strong>${fmt(aObs, 1)} m²</strong></div>
    <div class="telemetry-box"><span>EROM Q_E / V_E</span><strong>${fmt(isEh ? s.qe : s.qe, 1)} m³/s · ${fmt(isEh ? s.ve : s.ve, 2)} m/s</strong></div>
    <div class="telemetry-box"><span>Bed D₅₀ / Aridity</span><strong>${fmt(s.d50, 1)} mm · AI ${fmt(s.ai, 2)}</strong></div>
    <div class="telemetry-box"><span>ADCP / Stage Obs</span><strong>${isEh ? "USACE Bathy" : `${s.ns} ADCP · ${s.nh} Flood`}</strong></div>
    <div class="telemetry-box"><span>Obs Shape r / V</span><strong>r = ${fmt(rObs, 2)} · ${vObs ? `${fmt(vObs, 2)} m/s` : "—"}</strong></div>
  `;

  renderXsPlot();
  renderRatingSimulator();
  renderSiteModelTable();
}

function renderXsPlot() {
  const svg = document.getElementById("xs-svg");
  const s = state.selectedSite;
  if (!svg || !s) return;
  svg.innerHTML = "";

  const isEh = state.selectedIsEhydro;
  const [wObs, dbarObs, dmaxObs, , , rObs] = s.obs;

  // Gather active models + Observed to determine scales
  const profiles = [
    { name: "Observed", W: wObs, dbar: dbarObs, dmax: dmaxObs, r: rObs || 2.0, color: "#0f172a", dashed: false, lw: 2.8 },
  ];

  for (const m of HEADLINE_MODELS) {
    if (!state.activeModels.has(m)) continue;
    const entry = getSiteModelEntry(s, isEh, m);
    if (!entry) continue;
    profiles.push({
      name: m,
      W: entry[0],
      dbar: entry[1],
      dmax: entry[2],
      A: entry[3],
      r: entry[4] || 2.0,
      n: entry[5],
      uq: entry.length >= 10 ? { wLo: entry[6], wHi: entry[7], dLo: entry[8], dHi: entry[9] } : null,
      color: MODEL_COLORS[m],
      dashed: m !== "M5" && m !== "M2-XGB",
      lw: m === "M5" || m === "M2-XGB" ? 2.3 : 1.7,
    });
  }

  const maxW = Math.max(...profiles.map((p) => (p.uq ? Math.max(p.W, p.uq.wHi * 0.85) : p.W))) * 1.12;
  const maxD = Math.max(...profiles.map((p) => (p.uq ? Math.max(p.dmax, p.uq.dHi * 1.25) : p.dmax))) * 1.18;

  const pad = { l: 54, r: 22, t: 24, b: 38 };
  const W_px = 680 - pad.l - pad.r;
  const H_px = 310 - pad.t - pad.b;

  const xToPx = (x_m) => pad.l + ((x_m + maxW / 2) / maxW) * W_px;
  const zToPx = (depth_m) => pad.t + (depth_m / maxD) * H_px; // 0 = water surface, maxD = bottom

  // Water fill for Observed channel
  const obsPts = [];
  const N = 64;
  for (let i = 0; i <= N; i++) {
    const xi = -1 + (2 * i) / N;
    const x = (xi * wObs) / 2;
    const depth = dmaxObs * (1 - Math.pow(Math.abs(xi), rObs || 2.0));
    obsPts.push(`${xToPx(x).toFixed(1)},${zToPx(depth).toFixed(1)}`);
  }
  const waterPoly = `${xToPx(-wObs / 2).toFixed(1)},${zToPx(0).toFixed(1)} ${obsPts.join(" ")} ${xToPx(
    wObs / 2
  ).toFixed(1)},${zToPx(0).toFixed(1)}`;
  svg.appendChild(
    svgEl("polygon", {
      points: waterPoly,
      fill: "rgba(56, 189, 248, 0.14)",
    })
  );

  // Grid lines & axes
  for (let frac = 0; frac <= 1.001; frac += 0.25) {
    const dVal = frac * maxD;
    const y = zToPx(dVal);
    svg.appendChild(
      svgEl("line", {
        x1: pad.l,
        y1: y,
        x2: pad.l + W_px,
        y2: y,
        stroke: "rgba(148, 163, 184, 0.28)",
        "stroke-dasharray": frac === 0 ? "none" : "3,3",
      })
    );
    svg.appendChild(
      svgEl(
        "text",
        { x: pad.l - 7, y: y + 4, "text-anchor": "end", fill: "#64748b", "font-size": "10.5" },
        `−${dVal.toFixed(1)} m`
      )
    );
  }

  for (let frac = -0.5; frac <= 0.501; frac += 0.25) {
    const xVal = frac * maxW;
    const x = xToPx(xVal);
    svg.appendChild(
      svgEl(
        "text",
        { x, y: pad.t + H_px + 16, "text-anchor": "middle", fill: "#64748b", "font-size": "10.5" },
        `${xVal >= 0 ? "+" : ""}${xVal.toFixed(0)} m`
      )
    );
  }

  svg.appendChild(
    svgEl(
      "text",
      { x: pad.l + W_px / 2, y: 304, "text-anchor": "middle", fill: "#64748b", "font-size": "11", "font-weight": "600" },
      "Lateral Distance from Channel Centerline x (m)"
    )
  );

  // Water Surface Line (z = 0)
  svg.appendChild(
    svgEl("line", {
      x1: xToPx(-wObs / 2),
      y1: zToPx(0),
      x2: xToPx(wObs / 2),
      y2: zToPx(0),
      stroke: "#0284c7",
      "stroke-width": 1.6,
    })
  );

  // 90% UQ Shaded Bands for M5 and M2-MLP (if toggled)
  if (state.showUqBand) {
    for (const p of profiles) {
      if (!p.uq) continue;
      const ratio = p.dmax / Math.max(p.dbar, 1e-3);
      const dmaxLo = Math.min(p.uq.dLo * ratio, maxD * 0.98);
      const dmaxHi = Math.min(p.uq.dHi * ratio, maxD * 0.98);
      const wLo = Math.min(p.uq.wLo, maxW * 0.98);
      const wHi = Math.min(p.uq.wHi, maxW * 0.98);

      const upperPts = [];
      const lowerPts = [];
      for (let i = 0; i <= N; i++) {
        const xi = -1 + (2 * i) / N;
        const xOuter = (xi * wHi) / 2;
        const zOuter = dmaxHi * (1 - Math.pow(Math.abs(xi), p.r));
        const xInner = (xi * wLo) / 2;
        const zInner = dmaxLo * (1 - Math.pow(Math.abs(xi), p.r));
        upperPts.push(`${xToPx(xOuter).toFixed(1)},${zToPx(zOuter).toFixed(1)}`);
        lowerPts.unshift(`${xToPx(xInner).toFixed(1)},${zToPx(zInner).toFixed(1)}`);
      }
      svg.appendChild(
        svgEl("polygon", {
          points: `${upperPts.join(" ")} ${lowerPts.join(" ")}`,
          fill: p.color,
          "fill-opacity": "0.14",
        })
      );
    }
  }

  // Draw Cross-Section Bed Profiles
  for (const p of profiles) {
    const pts = [];
    for (let i = 0; i <= N; i++) {
      const xi = -1 + (2 * i) / N;
      const x = (xi * p.W) / 2;
      const depth = Math.min(p.dmax * (1 - Math.pow(Math.abs(xi), p.r)), maxD);
      pts.push(`${xToPx(x).toFixed(1)},${zToPx(depth).toFixed(1)}`);
    }
    svg.appendChild(
      svgEl("polyline", {
        points: pts.join(" "),
        fill: "none",
        stroke: p.color,
        "stroke-width": p.lw,
        "stroke-dasharray": p.dashed ? "5,3" : "none",
      })
    );
  }

  // Legend inside SVG top-left
  let legX = pad.l + 8;
  for (const p of profiles) {
    svg.appendChild(
      svgEl("line", {
        x1: legX,
        y1: 13,
        x2: legX + 18,
        y2: 13,
        stroke: p.color,
        "stroke-width": 2.4,
        "stroke-dasharray": p.dashed ? "4,2" : "none",
      })
    );
    const labelText = p.name === "Observed" ? `Obs (W=${fmt(p.W, 0)}m, d̄=${fmt(p.dbar, 2)}m)` : p.name;
    svg.appendChild(
      svgEl("text", { x: legX + 22, y: 17, fill: p.color, "font-size": "10.5", "font-weight": "700" }, labelText)
    );
    legX += p.name === "Observed" ? 168 : 66;
  }

  // Optional 3DEP Lidar Valley Transect Inset (top-right corner of XS SVG)
  if (state.show3depInset && !isEh && s.fp && Array.isArray(s.fp[2])) {
    const [lenM, z0, dzArr] = s.fp;
    const inW = 175;
    const inH = 72;
    const inX = pad.l + W_px - inW - 8;
    const inY = pad.t + 6;

    svg.appendChild(
      svgEl("rect", {
        x: inX,
        y: inY,
        width: inW,
        height: inH,
        rx: 4,
        fill: "rgba(15, 23, 42, 0.82)",
        stroke: "#475569",
      })
    );
    svg.appendChild(
      svgEl(
        "text",
        { x: inX + 6, y: inY + 13, fill: "#38bdf8", "font-size": "9.5", "font-weight": "700" },
        `3DEP Lidar Valley (${lenM} m · z₀=${z0}m)`
      )
    );

    const minDz = Math.min(...dzArr, -dmaxObs);
    const maxDz = Math.max(...dzArr, 4.0);
    const spanDz = Math.max(maxDz - minDz, 1.0);
    const xiCoords = state.conus.schema.fp_xi_coords;
    const fpPts = dzArr.map((dz, idx) => {
      const px = inX + 8 + ((xiCoords[idx] + 1) / 2) * (inW - 16);
      const py = inY + inH - 8 - ((dz - minDz) / spanDz) * (inH - 26);
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    });
    svg.appendChild(
      svgEl("polyline", {
        points: fpPts.join(" "),
        fill: "none",
        stroke: "#4ade80",
        "stroke-width": 1.6,
      })
    );
  }
}

/* ==========================================================================
   6. Live OPS-1 Stage–Discharge Rating Curve Simulator
   ========================================================================== */

function computeDingmanStageRise(dmax0, r, qRatio, gamma) {
  // Exact closed-form power-law Dingman-Manning rating exponent beta_M = (3r + 3) / (5r + 3)
  // With dynamic roughness n / n_a = (Q / Q_a)^gamma, effective discharge ratio is (Q / Q_a)^(1 + gamma)
  const betaM = (3 * r + 3) / (5 * r + 3);
  const qEff = Math.pow(Math.max(qRatio, 0.05), 1.0 + gamma);
  const lambda = Math.pow(qEff, betaM);
  return dmax0 * (lambda - 1.0);
}

function renderRatingSimulator() {
  const svg = document.getElementById("rating-svg");
  const readout = document.getElementById("sim-q-readout");
  const s = state.selectedSite;
  if (!svg || !s) return;
  svg.innerHTML = "";

  const isEh = state.selectedIsEhydro;
  const gammaMap = state.benchmarks.operational.ops3_diagnostics.dynamic_roughness.gamma_lodo_by_division || {};
  const divGamma = gammaMap[s.d || s.div] ?? -0.3246;
  const gamma = state.simRoughnessMode === "dyn" ? divGamma : 0.0;

  const [wObs, dbarObs, dmaxObs, , , rObs] = s.obs;
  const dhOracle = computeDingmanStageRise(dmaxObs, rObs || 2.0, state.simQRatio, gamma);
  readout.textContent = `Q / Q_a = ${state.simQRatio.toFixed(1)}× · ${
    state.simRoughnessMode === "dyn" ? `γ̂ = ${divGamma.toFixed(3)}` : "γ = 0 (Const. n)"
  } · Oracle Δh = ${dhOracle >= 0 ? "+" : ""}${dhOracle.toFixed(2)} m`;

  const curves = [
    { name: "Oracle (Obs Anchor)", dmax0: dmaxObs, r: rObs || 2.0, color: "#0f172a", dashed: false, lw: 2.5 },
  ];
  for (const m of ["M2-XGB", "M2-MLP", "M5", "M3", "M1"]) {
    if (!state.activeModels.has(m) && m !== "M3") continue;
    const entry = getSiteModelEntry(s, isEh, m);
    if (!entry) continue;
    curves.push({
      name: m,
      dmax0: entry[2],
      r: entry[4] || 2.0,
      color: MODEL_COLORS[m],
      dashed: m === "M3",
      lw: 1.9,
    });
  }

  const qMax = 10.0;
  let maxDh = 1.5;
  let minDh = -0.5;
  for (const c of curves) {
    maxDh = Math.max(maxDh, computeDingmanStageRise(c.dmax0, c.r, qMax, gamma));
    minDh = Math.min(minDh, computeDingmanStageRise(c.dmax0, c.r, 0.4, gamma));
  }
  maxDh *= 1.1;

  const pad = { l: 50, r: 20, t: 16, b: 30 };
  const W_px = 680 - pad.l - pad.r;
  const H_px = 185 - pad.t - pad.b;

  const qToX = (q) => pad.l + ((q - 0.4) / (qMax - 0.4)) * W_px;
  const dhToY = (dh) => pad.t + H_px - ((dh - minDh) / (maxDh - minDh)) * H_px;

  // Zero stage-change reference line (Q/Q_a = 1)
  const yZero = dhToY(0);
  svg.appendChild(
    svgEl("line", {
      x1: pad.l,
      y1: yZero,
      x2: pad.l + W_px,
      y2: yZero,
      stroke: "rgba(148, 163, 184, 0.45)",
      "stroke-dasharray": "3,3",
    })
  );

  // Selected Q/Q_a vertical cursor
  const xSel = qToX(state.simQRatio);
  svg.appendChild(
    svgEl("line", {
      x1: xSel,
      y1: pad.t,
      x2: xSel,
      y2: pad.t + H_px,
      stroke: "#D55E00",
      "stroke-width": 1.5,
      "stroke-dasharray": "4,2",
    })
  );

  // Y-axis labels
  for (let i = 0; i <= 3; i++) {
    const val = minDh + (i / 3) * (maxDh - minDh);
    const y = dhToY(val);
    svg.appendChild(
      svgEl(
        "text",
        { x: pad.l - 6, y: y + 3, "text-anchor": "end", fill: "#64748b", "font-size": "10" },
        `${val >= 0 ? "+" : ""}${val.toFixed(1)}m`
      )
    );
  }

  // X-axis labels
  for (const qTick of [1, 2, 4, 6, 8, 10]) {
    const x = qToX(qTick);
    svg.appendChild(
      svgEl("text", { x, y: pad.t + H_px + 14, "text-anchor": "middle", fill: "#64748b", "font-size": "10" }, `${qTick}×`)
    );
  }

  // Draw curves
  for (const c of curves) {
    const pts = [];
    for (let q = 0.4; q <= qMax + 0.01; q += 0.25) {
      const dh = computeDingmanStageRise(c.dmax0, c.r, q, gamma);
      pts.push(`${qToX(q).toFixed(1)},${dhToY(dh).toFixed(1)}`);
    }
    svg.appendChild(
      svgEl("polyline", {
        points: pts.join(" "),
        fill: "none",
        stroke: c.color,
        "stroke-width": c.lw,
        "stroke-dasharray": c.dashed ? "4,2" : "none",
      })
    );
    const dhAtQ = computeDingmanStageRise(c.dmax0, c.r, state.simQRatio, gamma);
    svg.appendChild(
      svgEl("circle", {
        cx: xSel,
        cy: dhToY(dhAtQ),
        r: 3.5,
        fill: c.color,
      })
    );
  }
}

function renderSiteModelTable() {
  const tbody = document.getElementById("site-model-tbody");
  const s = state.selectedSite;
  if (!tbody || !s) return;
  const isEh = state.selectedIsEhydro;
  const [wObs, dbarObs, dmaxObs, aObs, , rObs] = s.obs;
  const epsObs = Math.abs(Math.log(aObs / (wObs * dbarObs)));

  const rowsHtml = [
    `<tr class="is-best-row">
      <th scope="row">Observed (ADCP)</th>
      <td data-label="W (m)">${fmt(wObs, 2)}</td>
      <td data-label="d̄ (m)">${fmt(dbarObs, 3)}</td>
      <td data-label="d_max (m)">${fmt(dmaxObs, 3)}</td>
      <td data-label="A (m²)">${fmt(aObs, 2)}</td>
      <td data-label="r̂">${fmt(rObs, 2)}</td>
      <td data-label="n̂ / n_back">—</td>
      <td data-label="|ε_A|">${fmt(epsObs, 4)}</td>
      <td data-label="Geom RMSE">0.000</td>
      <td data-label="Stage FP (m)">—</td>
      <td data-label="Stage Dyn (m)">—</td>
    </tr>`,
  ];

  const winGeom = !isEh && s.w[0] ? s.w[0][0] : s.best_area_model;

  for (const m of HEADLINE_MODELS) {
    const entry = getSiteModelEntry(s, isEh, m);
    if (!entry) continue;
    const [W, dbar, dmax, A, rHat, nHat] = entry;
    const epsA = W && dbar && A ? Math.abs(Math.log(A / (W * dbar))) : null;
    const mIdx = ALL_TEN_MODELS.indexOf(m);
    const rmGeom = !isEh && s.rm[0] && mIdx >= 0 ? s.rm[0][mIdx] : null;
    const rmFp = !isEh && s.rm[1] && mIdx >= 0 ? s.rm[1][mIdx] : null;
    const rmDyn = !isEh && s.rm[2] && mIdx >= 0 ? s.rm[2][mIdx] : null;

    rowsHtml.push(`
      <tr class="${m === winGeom ? "is-best-row" : ""}">
        <th scope="row"><span class="chip-dot" style="--chip-color:${MODEL_COLORS[m]};display:inline-block;margin-inline-end:5px;"></span>${m}${m === winGeom ? " ★" : ""}</th>
        <td data-label="W (m)">${fmt(W, 2)}</td>
        <td data-label="d̄ (m)">${fmt(dbar, 3)}</td>
        <td data-label="d_max (m)">${fmt(dmax, 3)}</td>
        <td data-label="A (m²)">${fmt(A, 2)}</td>
        <td data-label="r̂">${fmt(rHat, 2)}</td>
        <td data-label="n̂ / n_back">${fmt(nHat, 4)}</td>
        <td data-label="|ε_A|">${m === "M5" || m === "M3" ? "0.0000" : fmt(epsA, 4)}</td>
        <td data-label="Geom RMSE">${fmt(rmGeom, 3)}</td>
        <td data-label="Stage FP (m)">${fmt(rmFp, 2)}</td>
        <td data-label="Stage Dyn (m)">${fmt(rmDyn, 2)}</td>
      </tr>
    `);
  }

  tbody.innerHTML = rowsHtml.join("");
}

/* ==========================================================================
   7. View 2: Confirmatory Geometry Benchmark (Event 1–3)
   ========================================================================== */

function initGeometryBenchmarkView() {
  const splitSel = document.getElementById("bench-split-select");
  const targetSel = document.getElementById("bench-target-select");

  const update = () => {
    renderBenchmarkChartAndTable(splitSel.value, targetSel.value);
  };
  splitSel.addEventListener("change", update);
  targetSel.addEventListener("change", update);
  update();

  renderHypothesesAndAblations();
}

function renderBenchmarkChartAndTable(split, target) {
  const matrix = state.benchmarks.metrics_matrix.filter(
    (r) => r.family === split && r.target === target
  );
  matrix.sort((a, b) => (a.metrics.rmse_log ?? 99) - (b.metrics.rmse_log ?? 99));

  const consModels = state.benchmarks.consistency.models || {};
  const contMap = (state.benchmarks.consistency.continuity || {})[split] || {};

  const tbody = document.getElementById("bench-metrics-tbody");
  tbody.innerHTML = matrix
    .map((row, idx) => {
      const m = row.model;
      const mt = row.metrics;
      const epsMed = contMap[m] ? contMap[m].median_abs_eps_A : m === "M5" || m === "M3" ? 0.0 : null;
      const inadN =
        consModels[m] && consModels[m][split] && consModels[m][split].nu_back_outside_bounds_share !== undefined
          ? consModels[m][split].nu_back_outside_bounds_share * 100
          : m === "M5" || m === "M4" || m === "M3"
          ? 0.0
          : null;
      return `
        <tr class="${idx === 0 ? "is-best-row" : ""}">
          <th scope="row"><span class="chip-dot" style="--chip-color:${MODEL_COLORS[m] || "#64748b"};display:inline-block;margin-inline-end:6px;"></span>${m}${idx === 0 ? " ★" : ""}</th>
          <td data-label="RMSE_log">${fmt(mt.rmse_log, 4)}</td>
          <td data-label="MdAPE (%)">${fmt(mt.mdape, 2)}</td>
          <td data-label="NSE_log">${fmt(mt.nse_log, 4)}</td>
          <td data-label="Bias_log">${fmt(mt.bias_log, 4)}</td>
          <td data-label="RMSE_raw">${fmt(mt.rmse_raw, 2)}</td>
          <td data-label="KGE">${fmt(mt.kge, 3)}</td>
          <td data-label="Median |ε_A|">${epsMed === 0 ? "0.0000" : fmt(epsMed, 4)}</td>
          <td data-label="Inadmissible n (%)">${inadN !== null ? `${fmt(inadN, 2)}%` : "—"}</td>
        </tr>
      `;
    })
    .join("");

  // Render SVG horizontal comparison bars
  const svg = document.getElementById("bench-bar-svg");
  svg.innerHTML = "";
  const maxRmse = Math.max(...matrix.map((r) => r.metrics.rmse_log || 0.5), 0.5) * 1.12;
  const pad = { l: 92, r: 70, t: 18, b: 22 };
  const W_px = 760 - pad.l - pad.r;
  const rowH = Math.min(24, Math.floor((290 - pad.t - pad.b) / Math.max(matrix.length, 1)));

  matrix.forEach((row, i) => {
    const y = pad.t + i * rowH;
    const val = row.metrics.rmse_log || 0;
    const barW = (val / maxRmse) * W_px;
    const col = MODEL_COLORS[row.model] || "#64748b";

    svg.appendChild(
      svgEl("text", { x: pad.l - 8, y: y + 14, "text-anchor": "end", fill: "#64748b", "font-size": "11", "font-weight": "700" }, row.model)
    );
    svg.appendChild(
      svgEl("rect", { x: pad.l, y: y + 3, width: Math.max(barW, 2), height: rowH - 6, rx: 3, fill: col })
    );
    svg.appendChild(
      svgEl("text", { x: pad.l + barW + 6, y: y + 14, fill: col, "font-size": "11", "font-weight": "700" }, `${val.toFixed(4)} (${fmt(row.metrics.mdape, 1)}%)`)
    );
  });
}

function renderHypothesesAndAblations() {
  const listEl = document.getElementById("hypotheses-cards-list");
  const items = [
    {
      id: "H1 (Regional Transfer S-LRO & S-IND Accuracy Contrast)",
      verdict: "FALSIFIED (Row 4)",
      supported: false,
      detail:
        "S-LRO ΔRMSE_log(d̄) = M5 − M2-XGB = +0.2502 (95% CI [+0.2262, +0.2765]); ΔRMSE_log(A) = +0.3094 ([+0.2728, +0.3520]). On S-IND (131 USACE locations), M5 wins Flow Area A (0.9495 vs. 0.9839; Δ = −0.0343 [−0.1588, +0.0746]).",
    },
    {
      id: "H2 (Headwater Q1 Bias Reduction on S-ID)",
      verdict: "FALSIFIED",
      supported: false,
      detail:
        "In smallest drainage-area quintile Q₁ (n = 798 sites), |B_d̄(M5)| − |B_d̄(M2-XGB)| = +0.0299 (95% CI [−0.0335, +0.0740]) on S-ID and +0.0753 on S-LRO.",
    },
    {
      id: "H3 (Out-of-Distribution Skill-Loss Reduction)",
      verdict: "FALSIFIED (Point < 0, CI Straddles 0)",
      supported: false,
      detail:
        "M5 loses less skill from S-ID to OOD than M2-XGB across S-LRO (Δ_H3 = −0.0138 [−0.0494, +0.0209]), S-TAIL-DA (−0.0485), and S-TAIL-SO (−0.2088 [−0.4357, +0.0151]), but 95% CIs include 0.",
    },
    {
      id: "H4 (Latent Manning n̂ & Dingman Shape r̂ Plausibility)",
      verdict: "3/4 Passed · D₅₀ Correlation Falsified",
      supported: false,
      detail:
        "Literature band π_lit = 92.65% ≥ 80% (S-LRO 95.19%), zero boundary saturation (π_ν = 0.13%, π_r = 0.00%), positive slope correlation ρ_S(n̂, S) = +0.4431, but bed grain-size ρ_S(n̂, D₅₀) = −0.0203.",
    },
    {
      id: "S2 (End-to-End IFT Through Φ vs. Two-Stage M6-NN)",
      verdict: "SUPPORTED (Bonferroni 98.75% CI)",
      supported: true,
      detail:
        "End-to-end differentiation through Φ (M5) beats two-stage M6-NN (same I_blind roughness routing) by −0.0996 log units on S-ID and −0.1016 ([−0.1110, −0.0930]) on S-LRO.",
    },
  ];

  listEl.innerHTML = items
    .map(
      (it) => `
    <div class="hyp-card">
      <div class="hyp-card-top">
        <span>${it.id}</span>
        <span class="verdict-pill ${it.supported ? "is-supported" : "is-falsified"}">${it.verdict}</span>
      </div>
      <p class="kpi-desc">${it.detail}</p>
    </div>`
    )
    .join("");

  // Ablations bar chart on S-LRO dbar
  const ablSvg = document.getElementById("ablation-svg");
  ablSvg.innerHTML = "";
  const ablRows = [
    { label: "M2-XGB (M* Unconstrained)", val: 0.4782, col: "#0072B2" },
    { label: "M2-MLP (Deep Ensemble)", val: 0.4866, col: "#56B4E9" },
    { label: "M5 + SA5 (Drop Floored Slope F_S=1)", val: 0.6067, col: "#009E73" },
    { label: "M5 + A4 (Oracle Measured Q_obs in Φ)", val: 0.6184, col: "#14B8A6" },
    { label: "M4 (Soft Penalty PINN)", val: 0.6864, col: "#E69F00" },
    { label: "M5 Primary (Hard Φ, I_blind)", val: 0.7284, col: "#D55E00" },
    { label: "M6-NN (Two-Stage Back-Calc n)", val: 0.8300, col: "#8B5CF6" },
    { label: "M3 (Physics Only, Strickler/Jarrett n)", val: 1.0156, col: "#CC79A7" },
  ];

  const pad = { l: 215, r: 58, t: 12, b: 16 };
  const W_px = 620 - pad.l - pad.r;
  const rowH = 28;
  ablRows.forEach((r, i) => {
    const y = pad.t + i * rowH;
    const bw = (r.val / 1.1) * W_px;
    ablSvg.appendChild(
      svgEl("text", { x: pad.l - 8, y: y + 16, "text-anchor": "end", fill: "#64748b", "font-size": "10.5", "font-weight": "600" }, r.label)
    );
    ablSvg.appendChild(
      svgEl("rect", { x: pad.l, y: y + 4, width: bw, height: 18, rx: 3, fill: r.col })
    );
    ablSvg.appendChild(
      svgEl("text", { x: pad.l + bw + 6, y: y + 16, fill: r.col, "font-size": "10.5", "font-weight": "700" }, r.val.toFixed(4))
    );
  });
}

/* ==========================================================================
   8. View 3: Operational Stage Routing & CCHE1D Readiness (OPS-1 v1–v3)
   ========================================================================== */

function initOperationalView() {
  const regSel = document.getElementById("ops-regime-select");
  const ccheSel = document.getElementById("cche-split-select");

  const renderOps = () => renderOpsStageChartAndTable(regSel.value);
  const renderCche = () => renderCche1dTable(ccheSel.value);

  regSel.addEventListener("change", renderOps);
  ccheSel.addEventListener("change", renderCche);

  renderOps();
  renderCche();
  renderOffAnchorTable();
}

function renderOpsStageChartAndTable(regimeMode) {
  const ops = state.benchmarks.operational;
  const v1Map = (ops.ops1_v1.primary_s_id || {}).stage_rmse_m || {};
  const v2FpMap = ((ops.ops2_floodplain.s_id || {}).floodplain || {}).stage_rmse_m || {};
  const slopeStrat =
    (((ops.ops3_diagnostics.v2_all_models || {})["S-ID"] || {}).slope_stratification || {}).floodplain || {};
  const notFsMap = (slopeStrat.not_F_S || {}).rmse_m || {};
  const fsMap = (slopeStrat.F_S || {}).rmse_m || {};

  const dynTable = (ops.ops3_diagnostics.dynamic_roughness || {}).table || [];
  const dynMap = {};
  const constOracleMap = {};
  for (const r of dynTable) {
    if (r.regime === "dynamic_n_lodo") dynMap[r.model] = r.high_flow_rmse_m;
    if (r.regime === "constant_n") constOracleMap[r.model] = r.high_flow_rmse_m;
  }

  const sharesGeom = {};
  for (const sh of ops.best_model_shares || []) {
    if (sh.criterion === "geom") sharesGeom[sh.model] = sh;
  }

  const tbody = document.getElementById("ops-stage-tbody");
  tbody.innerHTML = ALL_TEN_MODELS.map((m) => {
    const v1 = v1Map[m] ? v1Map[m].high : null;
    const v2 = v2FpMap[m] ? v2FpMap[m].high : null;
    const notFs = notFsMap[m];
    const fs = fsMap[m];
    const dyn = dynMap[m];
    const sh = sharesGeom[m] || {};
    const bestRow = m === "M2-XGB" || m === "M2-MLP" || m === "M4" || m === "M5";
    return `
      <tr class="${bestRow ? "is-best-row" : ""}">
        <th scope="row"><span class="chip-dot" style="--chip-color:${MODEL_COLORS[m]};display:inline-block;margin-inline-end:6px;"></span>${m}</th>
        <td data-label="v1 In-Channel (m)">${fmt(v1, 3)}</td>
        <td data-label="v2 3DEP Pooled (m)">${fmt(v2, 3)}</td>
        <td data-label="v2 Normal S > 10⁻⁵">${fmt(notFs, 3)}${m === "M4" ? " ★#1" : m === "M5" ? " ★#2" : ""}</td>
        <td data-label="v2 Floored S = 10⁻⁵">${fmt(fs, 3)}</td>
        <td data-label="v3 Dynamic-n LODO">${fmt(dyn, 3)}${m === "M2-MLP" ? " ★#1" : m === "M2-XGB" ? " ★#2" : ""}</td>
        <td data-label="Win Share (Geom)">${sh.win_share !== undefined ? `${(sh.win_share * 100).toFixed(1)}%` : "—"}</td>
        <td data-label="Near-Best ≤ 0.05">${sh.near_best_share !== undefined ? `${(sh.near_best_share * 100).toFixed(1)}%` : "—"}</td>
      </tr>
    `;
  }).join("");

  // Render paired bar chart in #ops-stage-svg
  const svg = document.getElementById("ops-stage-svg");
  svg.innerHTML = "";
  const modelsToPlot = ["M4", "M5", "M2-XGB", "M2-MLP", "M0", "M1", "M3"];
  let pairs = [];
  let legA = "";
  let legB = "";

  if (regimeMode === "slope_strat") {
    legA = "Normal Slope S > 10⁻⁵ (79.1% · M4 & M5 Win)";
    legB = "Floored Slope S = 10⁻⁵ (20.9% · Clamped)";
    pairs = modelsToPlot.map((m) => ({ m, a: notFsMap[m] || 0, b: fsMap[m] || 0 }));
  } else if (regimeMode === "const_vs_dyn") {
    legA = "Dynamic Roughness n ∝ Q^γ̂ (OPS-1 v3 LODO)";
    legB = "Constant Roughness γ = 0 (OPS-1 v1)";
    pairs = ["Oracle", ...modelsToPlot].map((m) => ({
      m,
      a: dynMap[m] || 0,
      b: constOracleMap[m] || 0,
    }));
  } else {
    legA = "OPS-1 v2: 3DEP Lidar Floodplain (m)";
    legB = "OPS-1 v1: In-Channel Power Law (m)";
    pairs = modelsToPlot.map((m) => ({
      m,
      a: v2FpMap[m] ? v2FpMap[m].high : 0,
      b: v1Map[m] ? v1Map[m].high : 0,
    }));
  }

  const maxV = Math.max(...pairs.flatMap((p) => [p.a, p.b]), 2.5) * 1.12;
  const pad = { l: 84, r: 58, t: 28, b: 20 };
  const W_px = 740 - pad.l - pad.r;
  const rowH = Math.floor((300 - pad.t - pad.b) / pairs.length);

  // Legend
  svg.appendChild(svgEl("rect", { x: pad.l, y: 6, width: 12, height: 12, rx: 2, fill: "#0F7C80" }));
  svg.appendChild(svgEl("text", { x: pad.l + 16, y: 16, fill: "#0F7C80", "font-size": "11", "font-weight": "700" }, legA));
  svg.appendChild(svgEl("rect", { x: pad.l + 310, y: 6, width: 12, height: 12, rx: 2, fill: "#D55E00" }));
  svg.appendChild(svgEl("text", { x: pad.l + 326, y: 16, fill: "#D55E00", "font-size": "11", "font-weight": "700" }, legB));

  pairs.forEach((p, i) => {
    const y = pad.t + i * rowH;
    const wA = (p.a / maxV) * W_px;
    const wB = (p.b / maxV) * W_px;
    svg.appendChild(
      svgEl("text", { x: pad.l - 8, y: y + 18, "text-anchor": "end", fill: "#64748b", "font-size": "11", "font-weight": "700" }, p.m)
    );
    svg.appendChild(svgEl("rect", { x: pad.l, y: y + 3, width: wA, height: 11, rx: 2, fill: "#0F7C80" }));
    svg.appendChild(
      svgEl("text", { x: pad.l + wA + 5, y: y + 12, fill: "#0F7C80", "font-size": "10", "font-weight": "700" }, `${p.a.toFixed(3)} m`)
    );
    svg.appendChild(svgEl("rect", { x: pad.l, y: y + 15, width: wB, height: 11, rx: 2, fill: "#D55E00", "fill-opacity": "0.8" }));
    svg.appendChild(
      svgEl("text", { x: pad.l + wB + 5, y: y + 24, fill: "#D55E00", "font-size": "10", "font-weight": "600" }, `${p.b.toFixed(3)} m`)
    );
  });
}

function renderCche1dTable(split) {
  const cche = (state.benchmarks.operational.ops3_diagnostics.cche1d_readiness || {}).splits || {};
  const rows = cche[split] || [];
  const sorted = [...rows].sort((a, b) => (a.rmse_log_R0 ?? 99) - (b.rmse_log_R0 ?? 99));
  const tbody = document.getElementById("cche-tbody");
  tbody.innerHTML = sorted
    .map(
      (r, idx) => `
      <tr class="${idx === 0 ? "is-best-row" : ""}">
        <th scope="row"><span class="chip-dot" style="--chip-color:${MODEL_COLORS[r.model] || "#0F7C80"};display:inline-block;margin-inline-end:6px;"></span>${r.model}${idx === 0 ? " ★" : ""}</th>
        <td data-label="RMSE_log(R_0)">${fmt(r.rmse_log_R0, 4)}</td>
        <td data-label="RMSE_raw(R_0) (m)">${fmt(r.rmse_raw_R0_m, 3)}</td>
        <td data-label="RMSE_log(Â_cons)">${fmt(r.rmse_log_A_cons, 4)}</td>
        <td data-label="RMSE_log(A_direct)">${fmt(r.rmse_log_A_direct, 4)}</td>
        <td data-label="Orig. |ε_A| > 0.05 (%)">${fmt((r.orig_continuity_violation_share || 0) * 100, 1)}% → 0.0%</td>
        <td data-label="P_99(|Δd_max|) (m)">${fmt(r.p99_abs_dmax_err_m, 2)} m</td>
      </tr>`
    )
    .join("");
}

function renderOffAnchorTable() {
  const oa = (state.benchmarks.operational.ops3_diagnostics.off_anchor_states || {}).table || [];
  const sorted = [...oa].sort((a, b) => (a.rmse_ln_dbar ?? 99) - (b.rmse_ln_dbar ?? 99));
  const tbody = document.getElementById("offanchor-tbody");
  tbody.innerHTML = sorted
    .map(
      (r, idx) => `
      <tr class="${idx < 2 ? "is-best-row" : ""}">
        <th scope="row"><span class="chip-dot" style="--chip-color:${MODEL_COLORS[r.model] || "#64748b"};display:inline-block;margin-inline-end:6px;"></span>${r.model}${idx === 0 ? " ★#1" : idx === 1 ? " ★#2" : ""}</th>
        <td data-label="RMSE(ln d̄(Q_i))">${fmt(r.rmse_ln_dbar, 4)}</td>
        <td data-label="RMSE(ln d_max(Q_i))">${fmt(r.rmse_ln_dmax, 4)}</td>
        <td data-label="RMSE(ln W(Q_i))">${fmt(r.rmse_ln_W, 4)}</td>
        <td data-label="RMSE(ln A(Q_i))">${fmt(r.rmse_ln_A, 4)}</td>
        <td data-label="High-Flow Raw d_max(Q_i) (m)">${fmt(r.high_flow_rmse_raw_dmax_m, 3)} m</td>
      </tr>`
    )
    .join("");
}

/* ==========================================================================
   9. View 4: Data Availability, QA/QC Funnel & Provenance + Ledger Modal
   ========================================================================== */

function initProvenanceView() {
  const funnelEl = document.getElementById("qc-funnel-list");
  const steps = [
    {
      stage: "1. Raw USGS HYDROSWOT Acquisition (acquire → unit_map)",
      count: "223,022 ADCP / current-meter records · 10,081 unique USGS stations",
      desc: "Deterministic SI unit conversion (ft → m, cfs → m³/s); SHA-256 verified against USGS ScienceBase release (doi:10.5066/F7D798H6).",
    },
    {
      stage: "2. Row-Wise Physical QA/QC Cascade (R1–R11 in src/data/qc.py)",
      count: "64,956 valid complete channel geometry measurements retained",
      desc: "Enforces positive W, d̄, d_max, A, Q, V; continuity closure |ln(A/(W·d̄))| ≤ ln(1.05); shape bounds r_obs ∈ [0.5, 10.0]; and velocity-area consistency.",
    },
    {
      stage: "3. Topological NHDPlusV2.1 Flowline Linking & Mean-Flow Anchor Selection (R12–R14)",
      count: "4,205 linked gauges → 3,988 Primary CONUS Benchmark Sites (~F_DA)",
      desc: "Linked via GageLoc + 500 m snap fallback and drainage-area agreement |ln(DA_site/DA_NHD)| ≤ ln(1.25); anchor row = argmin |ln(Q_obs/Q_E)|.",
    },
    {
      stage: "4. Operational Stage-Response & USGS 3DEP Lidar Floodplain Transects (OPS-1 v1–v3)",
      count: "75,343 off-anchor stage obs · 3,282 3DEP 10 m Lidar transects · 31,367 dynamic ADCP states",
      desc: "1,001-point perpendicular 3DEP terrain profiles (NAVD88/VERTCON 3.0 aligned) + leave-one-division-out Leopold–Maddock dynamic resistance.",
    },
    {
      stage: "5. Independent USACE eHydro Bathymetric Validation Cohort (S-IND)",
      count: "131 surveyed navigation locations across 18 rivers (33 single-COMID UQ locations)",
      desc: "Completely independent measurement agency and multibeam/singlebeam hydrographic survey modality (> 1 km along network from any training gauge).",
    },
  ];

  funnelEl.innerHTML = steps
    .map(
      (s) => `
    <div class="hyp-card">
      <div class="hyp-card-top">
        <span>${s.stage}</span>
        <span class="badge-pill is-good">${s.count}</span>
      </div>
      <p class="kpi-desc">${s.desc}</p>
    </div>`
    )
    .join("");

  const ev1 = state.benchmarks.event1_audit || {};
  const ev1El = document.getElementById("event1-audit-cards");
  ev1El.innerHTML = `
    <div class="telemetry-box"><span>Audit Classification</span><strong>${ev1.label || "optimistic"}</strong></div>
    <div class="telemetry-box"><span>Median Nearest Gauge</span><strong>${fmt(ev1.d_nn_median_km, 2)} km</strong></div>
    <div class="telemetry-box"><span>Variogram Range ρ̂</span><strong>${fmt(ev1.rho_hat_km, 2)} km</strong></div>
    <div class="telemetry-box"><span>Strongly Nested Ψ ≥ 0.5</span><strong>42.13% of sites</strong></div>
  `;

  const provRows = [
    ["HYDROSWOT", "USGS (Canova et al., 2016)", "n = 3,988 CONUS sites · 75,343 OPS-1 stage rows · 9,673 S-RND rows", "CC0-1.0", "https://doi.org/10.5066/F7D798H6"],
    ["NHDPlusV2.1", "U.S. EPA & USGS (McKay et al., 2012)", "1:100k hydrography (COMID, Q_E, V_E, S, F_S, Stream Order)", "CC0-1.0", "https://doi.org/10.5066/F7P84B24"],
    ["EPA StreamCat", "U.S. EPA (Hill et al., 2016)", "Catchment climate, soils, lithology, land cover, dam regulation", "CC0-1.0", "https://doi.org/10.1111/1752-1688.12372"],
    ["CONUS Bed D₅₀", "Abeshu et al. (2022)", "Reach-scale median bed-material grain size D₅₀ (mm)", "CC-BY-4.0", "https://doi.org/10.5281/zenodo.5550700"],
    ["Global Aridity v3.1", "Zomer et al. (2022)", "High-resolution aridity index (AI) & PET covariates", "CC-BY-4.0", "https://doi.org/10.6084/m9.figshare.7504448.v5"],
    ["USGS 3DEP (10 m DEM)", "USGS National Map (OPS-1 v2)", "3,282 perpendicular floodplain transects (1,001 pts each, NAVD88)", "CC0-1.0", "https://doi.org/10.3133/fs20143083"],
    ["Physiographic Divisions", "Fenneman & Johnson (1946) / USGS", "8 CONUS physiographic divisions defining S-LRO regional holdouts", "CC0-1.0", "https://water.usgs.gov/GIS/metadata/usgswrd/XML/physio.xml"],
    ["USACE eHydro (S-IND)", "U.S. Army Corps of Engineers", "131 independent surveyed navigation locations across 18 rivers", "CC0-1.0", "https://navigation.usace.army.mil/Survey/Hydro"],
    ["HyG / US-SWD (S1)", "Enzminger et al. (2023, 2024)", "Derived Manning roughness consistency benchmark (2,887 matched sites)", "CC-BY-4.0", "https://doi.org/10.5281/zenodo.10035344"],
  ];

  const provTbody = document.getElementById("prov-tbody");
  provTbody.innerHTML = provRows
    .map(
      ([name, agency, role, lic, url]) => `
    <tr>
      <th scope="row">${name}</th>
      <td data-label="Agency / Source">${agency}</td>
      <td data-label="Benchmark Role & Coverage">${role}</td>
      <td data-label="License"><span class="badge-pill">${lic}</span></td>
      <td data-label="DOI / Persistent Link"><a href="${url}" target="_blank" rel="noopener noreferrer">${url.replace("https://", "")}</a></td>
    </tr>`
    )
    .join("");
}

function populateLedgerDialog() {
  const ledger = state.benchmarks.data_provenance.scoring_ledger || {};
  const events = ledger.events || [];
  const meta = state.benchmarks.meta || {};

  const pillsEl = document.getElementById("ledger-meta-pills");
  pillsEl.innerHTML = `
    <span class="badge-pill is-good">Run ID: ${meta.run_id || "p8-da9b7303cb84"}</span>
    <span class="badge-pill is-good">7/7 Events git_dirty: false</span>
    <span class="badge-pill">params.yaml SHA-256: ${(meta.params_sha256 || "").slice(0, 12)}…</span>
    <span class="badge-pill">Phase 8 Merkle Leaves: 51,508</span>
  `;

  const tbody = document.getElementById("ledger-tbody");
  tbody.innerHTML = events
    .map(
      (ev, i) => `
    <tr>
      <th scope="row">${i + 1}</th>
      <td data-label="Event"><code>${ev.event}</code></td>
      <td data-label="Timestamp (UTC)">${ev.timestamp_utc || ev.ts_utc || "—"}</td>
      <td data-label="Git Commit"><code>${(ev.git_head || ev.commit || "").slice(0, 10)}</code></td>
      <td data-label="Clean Tree"><span class="badge-pill is-good">git_dirty: ${String(ev.git_dirty)}</span></td>
      <td data-label="Prev Hash → Entry SHA-256"><code>${(ev.prev_sha256 || "GENESIS").slice(0, 10)}… → ${(ev.entry_sha256 || ev.sha256 || "").slice(0, 12)}…</code></td>
    </tr>`
    )
    .join("");
}

document.addEventListener("DOMContentLoaded", () => {
  initApp().catch((err) => {
    console.error("Failed to initialize WebGIS:", err);
  });
});
