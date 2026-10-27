import xml.etree.ElementTree as ET
import pandas as pd

NS = {"coah": "http://sinai.ujaen.es/coah"}

def cargar(ruta):
    raiz = ET.parse(ruta).getroot()
    filas = []
    for r in raiz.findall("coah:hotel_review", NS):
        filas.append({
            "id": int(r.findtext("coah:id", namespaces=NS)),
            "rank": int(r.findtext("coah:rank", namespaces=NS)),
            "titulo": (r.findtext("coah:abstract", namespaces=NS) or "").strip(),
            "texto": (r.findtext("coah:review", namespaces=NS) or "").strip(),
        })
    df = pd.DataFrame(filas)
    # 1-2 negativo, 3 neutro, 4-5 positivo
    df["sentimiento"] = df["rank"].map(lambda x: "negativo" if x <= 2 else ("neutro" if x == 3 else "positivo"))
    return df

if __name__ == "__main__":
    df = cargar("data/corpus_coah.xml")
    print("Filas:", len(df))
    print("Faltantes:\n", df.isna().sum().to_string())
    print("Vacíos texto:", (df.texto == "").sum(), "| vacíos título:", (df.titulo == "").sum())
    print("IDs duplicados:", df.id.duplicated().sum(), "| textos duplicados:", df.texto.duplicated().sum())
    print("\nPor rank:\n", df["rank"].value_counts().sort_index().to_string())
    print("\nPor sentimiento:\n", df["sentimiento"].value_counts().to_string())
    df["palabras"] = df.texto.str.split().str.len()
    print("\nPalabras por reseña:\n", df.palabras.describe().round(1).to_string())
    df.drop(columns="palabras").to_csv("data/coah.csv", index=False)
