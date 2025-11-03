"""
Data Advisor v16.0 - Professional Edition
Run: python app.py
"""
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import numpy as np
import io
import logging
import os
from groq import Groq
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
# PDF Libraries
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import plotly.graph_objects as go
from io import BytesIO

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Data Advisor API v16.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_file(content: bytes, filename: str) -> pd.DataFrame:
    """Load CSV or Excel met encoding fallback."""
    ext = filename.split('.')[-1].lower()
    try:
        if ext == 'csv':
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    return pd.read_csv(io.BytesIO(content), encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not decode CSV")
        elif ext in ['xlsx', 'xls']:
            return pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported: {ext}")
    except Exception as e:
        logger.error(f"Load error: {e}")
        raise ValueError(f"Load error: {e}")

def safe_json_convert(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime naar string voor JSON."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

def build_groq_prompt(df: pd.DataFrame) -> str:
    """Uitgebreide professionele Nederlandse analyse prompt voor Groq."""
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    missing_total = int(df.isnull().sum().sum())
    missing_pct = (missing_total / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 0
    
    # Uitgebreide statistieken met variatie-analyse
    stats_lines = []
    outlier_info = []
    if numeric:
        for col in numeric[:5]:  # Meer kolommen
            col_data = df[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                std_val = col_data.std()
                cv = (std_val / mean_val * 100) if mean_val != 0 else 0  # Coefficient of Variation
                
                stats_lines.append(
                    f"• {col}: μ={mean_val:.2f}, σ={std_val:.2f}, "
                    f"CV={cv:.1f}%, bereik=[{col_data.min():.2f}, {col_data.max():.2f}]"
                )
                
                # Detecteer outliers (z-score > 3)
                if std_val > 0:
                    z_scores = np.abs((col_data - mean_val) / std_val)
                    outlier_count = (z_scores > 3).sum()
                    if outlier_count > 0:
                        outlier_pct = (outlier_count / len(col_data) * 100)
                        outlier_info.append(f"• {col}: {outlier_count} outliers ({outlier_pct:.1f}%)")
    
    # Sterke correlaties met business context
    corr_lines = []
    if len(numeric) > 1:
        corr_matrix = df[numeric].corr().abs()
        correlations = []
        for i in range(len(numeric)):
            for j in range(i+1, len(numeric)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val) and corr_val > 0.3:  # Lagere drempel voor meer inzichten
                    correlations.append((numeric[i], numeric[j], corr_val))
        
        correlations.sort(key=lambda x: x[2], reverse=True)
        for col1, col2, corr_val in correlations[:5]:
            strength = "zeer sterk" if corr_val > 0.8 else "sterk" if corr_val > 0.6 else "matig"
            corr_lines.append(f"• {col1} ↔ {col2}: r={corr_val:.2f} ({strength})")
    
    # Categorische analyse
    cat_info = []
    if categorical:
        for col in categorical[:3]:
            unique_count = df[col].nunique()
            top_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
            top_freq = (df[col] == top_value).sum()
            top_pct = (top_freq / len(df) * 100)
            cat_info.append(
                f"• {col}: {unique_count} categorieën, "
                f"meest frequent '{top_value}' ({top_pct:.1f}%)"
            )
    
    # Missing value details
    missing_details = []
    if missing_total > 0:
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_col_pct = (missing_count / len(df) * 100)
                missing_details.append(f"• {col}: {missing_count} ({missing_col_pct:.1f}%)")
    
    return f"""Je bent een ervaren **senior data-analist**. Analyseer deze dataset grondig en lever actionable insights in het Nederlands.

═══════════════════════════════════════════════════════════════════
📊 DATASET PROFIEL
═══════════════════════════════════════════════════════════════════
Volume: {len(df):,} records × {len(df.columns)} variabelen
Numerieke features: {', '.join(numeric[:6]) if numeric else 'geen'}{'...' if len(numeric) > 6 else ''}
Categorische features: {', '.join(categorical[:4]) if categorical else 'geen'}{'...' if len(categorical) > 4 else ''}
Data compleetheid: {100-missing_pct:.1f}% ({missing_total:,} missing values)

📈 NUMERIEKE STATISTIEKEN (met variatiecoëfficiënt):
{chr(10).join(stats_lines) if stats_lines else 'Geen numerieke data'}

{f'''🚨 OUTLIERS GEDETECTEERD:
{chr(10).join(outlier_info[:4])}
''' if outlier_info else ''}

🔗 CORRELATIES (top 5):
{chr(10).join(corr_lines) if corr_lines else '• Geen sterke correlaties gevonden'}

📂 CATEGORISCHE VERDELING:
{chr(10).join(cat_info) if cat_info else 'Geen categorische variabelen'}

{f'''❌ MISSING VALUES PER KOLOM:
{chr(10).join(missing_details[:5])}
''' if missing_details else '✅ Geen missing values'}

═══════════════════════════════════════════════════════════════════
🎯 ANALYSE-OPDRACHT (strikte richtlijnen)
═══════════════════════════════════════════════════════════════════

PARAMETERS:
• Taal: Nederlands
• Lengte: 4 uitgebreide bullets (elk 2-3 zinnen)
• Stijl: Zakelijk, analytisch, direct bruikbaar voor besluitvorming
• Focus: Concrete, meetbare inzichten
• Vermijd: Vage uitspraken zoals "de data lijkt goed" of "meer onderzoek nodig"
• Na elke bullet een enter.
• Ik wil alle 4 de bullets dus: Kernbevinding, Datakwaliteit, Aanbeveling, Business Waarde

VERPLICHTE STRUCTUUR:

1. KERNBEVINDING (belangrijkste patroon/trend):
   → Noem de meest opvallende correlatie, trend of afwijking met exacte cijfers
   → Verklaar waarom dit belangrijk is voor het bedrijf
   → Voorbeeld: "Unit Price en Total Revenue correleren 0.93, wat betekent dat een prijsverhoging van 10% direct €X extra omzet kan opleveren"

2. DATAKWALITEIT (concrete issues + impact):
   → Als missing values: geef percentage + welke kolommen kritiek zijn
   → Als outliers: specificeer aantal + mogelijke oorzaak
   → Geef ALTIJD een praktisch verbeteradvies (bijv. "verwijder rijen met >20% missing")
   → Als data compleet is: benoem welke voorbereidingsstappen wél nodig zijn (normalisatie, encoding, etc.)

3. AANBEVELING (concrete vervolgstappen):
   → Koppel ELKE aanbeveling aan een analytisch doel:
     • Segmentatie-analyse → identificeer winstgevende klantsegmenten
     • Predictief model → voorspel churn/revenue/conversie
     • A/B testing → optimaliseer prijzen/campagnes
   → Geef prioriteit: "Start met X, daarna Y"
   → Voorbeeld: "1) Train een regressiemodel op price→revenue (verwacht R²>0.85). 2) Segmenteer klanten op basis van purchase frequency"

4. BUSINESSWAARDE (ROI-impact):
   → Formuleer één businessgerichte conclusie:
     • Winstoptimalisatie: "Prijsmodel kan omzet verhogen met X%"
     • Klantretentie: "Early churn detection kan €X per jaar besparen"
     • Procesverbetering: "Voorraadoptimalisatie reduceert kosten met X%"
   → Link data-inzichten aan KPI's (omzet, churn, conversie, kosten)
   → Wees specifiek: noem verwachte impact in percentages of bedragen waar mogelijk

BELANGRIJK:
• Begin DIRECT met punt 1 (geen inleidende tekst)
• Gebruik bullets (•) voor structuur
• Gebruik meetbare termen: percentages, correlaties, aantallen
• Baseer conclusies ALLEEN op zichtbare data
• Vermijd technisch jargon over statistiek
• Elke zin moet actionable zijn

Begin nu direct met de analyse:"""

@app.get("/")
def root():
    return {
        "status": "online",
        "version": "16.0",
        "ai": "Groq (llama-3.3-70b)",
        "endpoints": ["/upload", "/dashboard", "/train", "/insights", "/generate-pdf", "/health"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "16.0",
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload en analyseer bestand."""
    try:
        content = await file.read()
        df = load_file(content, file.filename)
        df = safe_json_convert(df)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        info = {
            "filename": file.filename,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing": {col: int(df[col].isnull().sum()) for col in df.columns},
            "missing_total": int(df.isnull().sum().sum()),
            "preview": df.head(10).fillna("").to_dict('records'),
            "numeric_stats": {}
        }
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                info["numeric_stats"][col] = {
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "count": int(len(col_data))
                }
        
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().fillna(0)
            info["correlation"] = {
                "columns": numeric_cols,
                "matrix": corr.values.tolist()
            }
        
        return JSONResponse(content=info)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/insights")
async def insights(file: UploadFile = File(...)):
    """Genereer AI insights met Groq."""
    try:
        logger.info("🔍 Generating enhanced AI insights...")
        content = await file.read()
        df = load_file(content, file.filename)
        df = safe_json_convert(df)
        
        # Sample grote datasets (maar behoud representativiteit)
        if len(df) > 1000:
            df = df.sample(n=1000, random_state=42)
            logger.info(f"📊 Sampled to 1000 rows for analysis")
        
        prompt = build_groq_prompt(df)
        
        # Verhoogde max_tokens voor uitgebreidere analyses
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": "Je bent een ervaren senior data-analist met 15 jaar ervaring. Je analyses zijn altijd concreet, meetbaar en direct bruikbaar voor strategische beslissingen. Je vermijdt vage taal en geeft altijd prioriteiten aan."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.8,  # Iets hoger voor meer creativiteit
            max_tokens=800,   # Verhoogd van 400 naar 800 voor uitgebreidere analyses
            top_p=0.95        # Hogere diversiteit
        )
        
        insights_text = chat_completion.choices[0].message.content.strip()
        
        logger.info(f"✅ Generated {len(insights_text)} chars of insights")
        
        return JSONResponse(content={
            "insights": insights_text,
            "model": "Groq (llama-3.3-70b-versatile)",
            "rows_analyzed": len(df),
            "tokens_used": chat_completion.usage.total_tokens,
            "prompt_length": len(prompt)
        })
        
    except Exception as e:
        logger.error(f"❌ Groq error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Groq error: {str(e)[:200]}")

@app.post("/dashboard")
async def dashboard(file: UploadFile = File(...)):
    """Genereer dashboard met visualisaties."""
    try:
        content = await file.read()
        df = load_file(content, file.filename)
        df = safe_json_convert(df)
        
        charts = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Histogrammen
        for col in numeric_cols[:4]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                charts.append({
                    "type": "histogram",
                    "title": f"Distributie: {col}",
                    "data": [{
                        "x": col_data.tolist(),
                        "type": "histogram",
                        "marker": {"color": "#4A5568"},
                        "name": col
                    }],
                    "layout": {"xaxis": {"title": col}, "yaxis": {"title": "Frequentie"}}
                })
        
        # Box plots
        if len(numeric_cols) > 0:
            box_data = []
            for col in numeric_cols[:5]:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    box_data.append({"y": col_data.tolist(), "type": "box", "name": col})
            if box_data:
                charts.append({"type": "box", "title": "Box Plot Vergelijking", "data": box_data})
        
        # Correlatie heatmap
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().fillna(0)
            charts.append({
                "type": "heatmap",
                "title": "Correlatie Matrix",
                "data": [{
                    "z": corr_matrix.values.tolist(),
                    "x": numeric_cols,
                    "y": numeric_cols,
                    "type": "heatmap",
                    "colorscale": "Blues"
                }]
            })
        
        result = {
            "charts": charts,
            "metrics": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "missing_values": int(df.isnull().sum().sum()),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "total_charts": len(charts)
            }
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-pdf")
async def generate_pdf(file: UploadFile = File(...)):
    """Genereer compleet PDF rapport met alle analyses."""
    try:
        logger.info("📄 Generating complete PDF report...")
        content = await file.read()
        df = load_file(content, file.filename)
        df = safe_json_convert(df)
        
        # Sample grote datasets
        original_size = len(df)
        if len(df) > 500:
            df = df.sample(n=500, random_state=42)
        
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=landscape(A4),
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30
        )
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2D3748'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2D3748'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#4A5568'),
            spaceAfter=8,
            leading=14
        )
        
        story = []
        
        # ==================== PAGINA 1: OVERZICHT ====================
        story.append(Paragraph("Data Advisor - Dashboard Rapport", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Dataset Info
        story.append(Paragraph("Dataset Overzicht", heading_style))
        info_data = [
            ['Eigenschap', 'Waarde'],
            ['Bestand', file.filename],
            ['Totaal Records', f"{original_size:,}"],
            ['Geanalyseerde Records', f"{len(df):,}"],
            ['Aantal Kolommen', str(len(df.columns))],
            ['Missing Values', f"{int(df.isnull().sum().sum()):,}"],
            ['Datum Rapport', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')]
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Kolommen Overzicht
        story.append(Paragraph("Kolommen in Dataset", heading_style))
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        columns_data = [['Type', 'Aantal', 'Kolommen']]
        if numeric_cols:
            columns_data.append(['Numeriek', len(numeric_cols), ', '.join(numeric_cols[:8]) + ('...' if len(numeric_cols) > 8 else '')])
        if categorical_cols:
            columns_data.append(['Categorisch', len(categorical_cols), ', '.join(categorical_cols[:8]) + ('...' if len(categorical_cols) > 8 else '')])
        
        columns_table = Table(columns_data, colWidths=[1.5*inch, 1*inch, 4.5*inch])
        columns_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(columns_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Numerieke Statistieken
        if numeric_cols:
            story.append(Paragraph("Numerieke Statistieken", heading_style))
            stats_data = [['Kolom', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']]
            for col in numeric_cols[:10]:  # Top 10
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    stats_data.append([
                        col,
                        f"{col_data.mean():.2f}",
                        f"{col_data.median():.2f}",
                        f"{col_data.std():.2f}",
                        f"{col_data.min():.2f}",
                        f"{col_data.max():.2f}"
                    ])
            
            stats_table = Table(stats_data, colWidths=[1.8*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A5568')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(stats_table)
        
        # ==================== PAGINA 2: VISUALISATIES ====================
        story.append(PageBreak())
        story.append(Paragraph("Visualisaties", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Chart 1: Histogram (eerste numerieke kolom)
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            col_data = df[col].dropna()
            
            fig = go.Figure(data=[go.Histogram(
                x=col_data,
                marker_color='#4A5568',
                name=col,
                nbinsx=30
            )])
            fig.update_layout(
                title=f'Distributie: {col}',
                xaxis_title=col,
                yaxis_title='Frequentie',
                width=700,
                height=350,
                template='plotly_white',
                font=dict(size=10)
            )
            
            img_bytes = fig.to_image(format="png", width=700, height=350)
            img = Image(BytesIO(img_bytes), width=5.5*inch, height=2.8*inch)
            story.append(img)
            story.append(Spacer(1, 0.15*inch))
        
        # Chart 2: Box Plot (top 4 numerieke kolommen)
        if len(numeric_cols) >= 2:
            fig = go.Figure()
            for col in numeric_cols[:4]:
                col_data = df[col].dropna()
                fig.add_trace(go.Box(
                    y=col_data,
                    name=col,
                    marker_color='#4A5568',
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title='Box Plot Vergelijking (Outlier Detectie)',
                yaxis_title='Waarden',
                width=700,
                height=350,
                template='plotly_white',
                showlegend=True,
                font=dict(size=10)
            )
            
            img_bytes = fig.to_image(format="png", width=700, height=350)
            img = Image(BytesIO(img_bytes), width=5.5*inch, height=2.8*inch)
            story.append(img)
            story.append(Spacer(1, 0.15*inch))
        
        # Chart 3: Correlatie Heatmap
        if len(numeric_cols) > 1:
            story.append(PageBreak())
            story.append(Paragraph("Correlatie Analyse", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            corr_matrix = df[numeric_cols[:8]].corr()  # Max 8 kolommen voor leesbaarheid
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='Blues',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 9},
                colorbar=dict(title="Correlatie")
            ))
            
            fig.update_layout(
                title='Correlatie Matrix (Hoe sterk hangen variabelen samen?)',
                width=700,
                height=550,
                template='plotly_white',
                font=dict(size=9)
            )
            
            img_bytes = fig.to_image(format="png", width=700, height=550)
            img = Image(BytesIO(img_bytes), width=5.5*inch, height=4.3*inch)
            story.append(img)
        
        # ==================== PAGINA 3: AI ANALYSE & ADVIES ====================
        story.append(PageBreak())
        story.append(Paragraph("AI Analyse & Strategisch Advies", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Genereer AI insights
        try:
            logger.info("🤖 Generating AI insights for PDF...")
            prompt = build_groq_prompt(df)
            
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Je bent een senior data consultant. Geef een professionele analyse in Nederlands met concrete business aanbevelingen. Gebruik duidelijke structuur met bullets en kopjes."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.8,
                max_tokens=800
            )
            
            insights_text = chat_completion.choices[0].message.content.strip()
            logger.info(f"✅ Generated {len(insights_text)} chars of insights")
            
        except Exception as e:
            logger.error(f"❌ AI insights error: {e}")
            insights_text = """**AI Analyse Niet Beschikbaar**

De AI-analyse kon niet worden gegenereerd. Controleer of de Groq API key correct is geconfigureerd.

**Handmatige Analyse Aanbevolen:**
• Analyseer correlaties tussen variabelen
• Controleer op outliers en missing values
• Identificeer trends en patronen
• Bepaal vervolgstappen voor data-analyse"""
        
        # Verwerk insights in paragrafen (respecteer formatting)
        for line in insights_text.split('\n'):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.05*inch))
                continue
            
            # Detecteer kopjes (** of ###)
            if line.startswith('**') and line.endswith('**'):
                # Bold heading
                heading_text = line.strip('*').strip()
                story.append(Paragraph(
                    f"<b>{heading_text}</b>",
                    ParagraphStyle('SubHeading', parent=body_style, fontSize=11, textColor=colors.HexColor('#2D3748'), spaceAfter=6)
                ))
            elif line.startswith('#'):
                # Markdown heading
                heading_text = line.lstrip('#').strip()
                story.append(Paragraph(
                    f"<b>{heading_text}</b>",
                    ParagraphStyle('SubHeading', parent=body_style, fontSize=11, textColor=colors.HexColor('#2D3748'), spaceAfter=6)
                ))
            elif line.startswith('•') or line.startswith('-'):
                # Bullet point
                bullet_text = line.lstrip('•-').strip()
                story.append(Paragraph(
                    f"• {bullet_text}",
                    ParagraphStyle('Bullet', parent=body_style, leftIndent=15, spaceAfter=4)
                ))
            else:
                # Normale paragraaf
                story.append(Paragraph(line, body_style))
        
        story.append(Spacer(1, 0.2*inch))
        
        # ==================== ADVIES SECTIE ====================
        story.append(Paragraph("Concrete Vervolgstappen", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Genereer advies op basis van data karakteristieken
        advice_items = []
        
        # Advies 1: Data kwaliteit
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        if missing_pct > 5:
            advice_items.append(
                f"<b>Data Cleaning:</b> De dataset bevat {missing_pct:.1f}% missing values. "
                f"Aanbeveling: Implementeer een imputation strategie (mediaan voor numeriek, modus voor categorisch) "
                f"of verwijder records met >20% missende data."
            )
        else:
            advice_items.append(
                "<b>Data Kwaliteit:</b> Data is grotendeels compleet. "
                "Voer een outlier-analyse uit met de Box Plots en overweeg robuuste modellen (Random Forest) die outliers kunnen handlen."
            )
        
        # Advies 2: Correlaties
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            max_corr = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).max().max()
            if max_corr > 0.7:
                advice_items.append(
                    f"<b>Multicollineariteit:</b> Er zijn sterke correlaties gedetecteerd (max: {max_corr:.2f}). "
                    f"Voor predictieve modellen: gebruik feature selection (PCA, Lasso) of train ensemble models die hier mee om kunnen gaan."
                )
            else:
                advice_items.append(
                    "<b>Feature Independence:</b> Variabelen zijn grotendeels onafhankelijk. "
                    "Dit is ideaal voor lineaire modellen en maakt interpretatie eenvoudiger."
                )
        
        # Advies 3: Modeling
        if len(numeric_cols) >= 3:
            advice_items.append(
                "<b>Machine Learning:</b> Train een Random Forest model om patronen te ontdekken. "
                "Start met de hoogst gecorreleerde variabele als target. "
                "Verwacht R² > 0.70 als correlaties sterk zijn. "
                "Gebruik 80/20 train-test split en cross-validatie voor robuustheid."
            )
        
        # Advies 4: Business impact
        advice_items.append(
            "<b>Implementatie:</b> Integreer inzichten in dashboards (Power BI, Tableau). "
            "Monitor key metrics maandelijks. "
            "A/B test strategieën voordat je organisatie-breed uitrolt. "
            "Documenteer bevindingen voor stakeholder buy-in."
        )
        
        for i, advice in enumerate(advice_items, 1):
            story.append(Paragraph(f"{i}. {advice}", body_style))
            story.append(Spacer(1, 0.08*inch))
        
        # ==================== FOOTER ====================
        story.append(Spacer(1, 0.4*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(
            f"Data Advisor v16.0 | Gegenereerd op {pd.Timestamp.now().strftime('%Y-%m-%d om %H:%M')} | Powered by AI",
            footer_style
        ))
        
        # Build PDF
        logger.info("📄 Building PDF document...")
        doc.build(story)
        pdf_buffer.seek(0)
        
        logger.info("✅ PDF generated successfully")
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=data_advisor_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ PDF generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF error: {str(e)[:200]}")

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target: str = Form(...),
    model_type: str = Form("regression"),
    test_size: float = Form(0.2),
    n_estimators: int = Form(100),
    max_depth: Optional[int] = Form(None)
):
    """Train ML model."""
    try:
        content = await file.read()
        df = load_file(content, file.filename)
        df = safe_json_convert(df)
        
        if target not in df.columns:
            raise ValueError(f"Column '{target}' not found")
        
        df = df.dropna(subset=[target])
        X = df.drop(columns=[target])
        y = df[target]
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols].copy()
        
        if len(X.columns) == 0:
            raise ValueError("No numeric features")
        
        X = X.fillna(X.median())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if model_type == "regression":
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            result = {
                "model": "Random Forest Regressor",
                "parameters": {"n_estimators": n_estimators, "max_depth": max_depth, "test_size": test_size},
                "metrics": {
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "R2": float(r2_score(y_test, y_pred)),
                    "MAE": float(np.mean(np.abs(y_test - y_pred)))
                },
                "feature_importance": [{"feature": col, "importance": float(imp)} for col, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:10]],
                "sample_predictions": {"actual": y_test.tolist()[:10], "predicted": y_pred.tolist()[:10]}
            }
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            result = {
                "model": "Random Forest Classifier",
                "parameters": {"n_estimators": n_estimators, "max_depth": max_depth, "test_size": test_size},
                "metrics": {"Accuracy": float(accuracy_score(y_test, y_pred))},
                "sample_predictions": {"actual": y_test.tolist()[:10], "predicted": y_pred.tolist()[:10]}
            }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("📊 DATA ADVISOR API v16.0 - PROFESSIONAL EDITION")
    print("="*60)
    print("🌐 Server:  http://localhost:8000")
    print("📚 Docs:    http://localhost:8000/docs")
    print("🤖 AI:      Groq (llama-3.3-70b-versatile)")
    print("📄 PDF:     Complete reports with AI analysis")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")