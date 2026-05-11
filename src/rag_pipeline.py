from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer

from data import DOCUMENTOS_MEDICOS

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


BI_ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_QUERY = "dor de cabeca latejante e luz incomodando"


def carregar_variaveis_locais() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return

    for linha in env_path.read_text(encoding="utf-8").splitlines():
        conteudo = linha.strip()
        if not conteudo or conteudo.startswith("#") or "=" not in conteudo:
            continue

        chave, valor = conteudo.split("=", 1)
        chave = chave.strip()
        valor = valor.strip().strip('"').strip("'")

        if chave and chave not in os.environ:
            os.environ[chave] = valor


@dataclass
class Busca:
    doc_id: str
    titulo: str
    texto: str
    score: float


class HydeGenerator:
    def __init__(self, model_name: str = "gpt-4.1-mini") -> None:
        self.model_name = model_name
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if api_key and OpenAI is not None:
            self.client = OpenAI(api_key=api_key)

    def gerar(self, query: str) -> str:
        if self.client is not None:
            prompt = (
                "Voce e um assistente clinico de indexacao semantica. "
                "Transforme a queixa coloquial abaixo em um pequeno paragrafo tecnico, "
                "com termos medicos, hipoteses diagnosticas e sinais associados. "
                "Nao escreva avisos. Produza apenas o documento hipotetico.\n\n"
                f"Queixa: {query}"
            )

            resposta = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=0.2,
            )
            return resposta.output_text.strip()

        return self._fallback(query)

    @staticmethod
    def _fallback(query: str) -> str:
        texto = query.lower()
        trocas = {
            "dor de cabeca": "cefaleia",
            "dor de cabeça": "cefaleia",
            "latejante": "pulsatil",
            "luz incomodando": "fotofobia",
            "luz incomoda": "fotofobia",
            "enjoo": "nauseas",
            "visao embacada": "reducao da acuidade visual",
            "visão embaçada": "reducao da acuidade visual",
            "tontura": "vertigem",
            "coracao acelerado": "taquicardia",
            "coração acelerado": "taquicardia",
            "falta de ar": "dispneia",
            "queimacao": "pirose",
            "queimação": "pirose",
            "ardendo ao urinar": "disuria",
            "muita vontade de urinar": "urgencia miccional",
        }

        ponte = texto
        for coloquial, tecnico in trocas.items():
            ponte = ponte.replace(coloquial, tecnico)

        return (
            "Documento hipotetico tecnico: paciente com "
            f"{ponte}, quadro descrito em terminologia medica e correlacionado com "
            "entidades clinicas provaveis, sintomas associados e linguagem de prontuario."
        )


class RagLaboratorio:
    def __init__(
        self,
        bi_encoder_model: str = BI_ENCODER_MODEL,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        self.documentos = DOCUMENTOS_MEDICOS
        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.hyde = HydeGenerator()
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = None

    def construir_indice(self) -> None:
        textos = [doc["texto"] for doc in self.documentos]
        embeddings = self.bi_encoder.encode(
            textos,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        dimensao = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimensao, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        index.add(embeddings)
        self.index = index

    def gerar_documento_hipotetico(self, query: str) -> str:
        return self.hyde.gerar(query)

    def recuperar(self, query: str, top_k: int = 10) -> List[Busca]:
        if self.index is None:
            raise RuntimeError("O indice ainda nao foi construido.")

        doc_hipotetico = self.gerar_documento_hipotetico(query)
        query_vector = self.bi_encoder.encode(
            [doc_hipotetico],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        scores, indices = self.index.search(query_vector, top_k)
        resultados = []

        for score, idx in zip(scores[0], indices[0]):
            doc = self.documentos[idx]
            resultados.append(
                Busca(
                    doc_id=doc["id"],
                    titulo=doc["titulo"],
                    texto=doc["texto"],
                    score=float(score),
                )
            )

        return resultados

    def reranquear(self, query: str, candidatos: List[Busca], top_k: int = 3) -> List[Busca]:
        consulta_expandida = f"{query}. Reformulacao tecnica: {self.gerar_documento_hipotetico(query)}"
        pares = [(consulta_expandida, f"{item.titulo}. {item.texto}") for item in candidatos]
        novos_scores = self.cross_encoder.predict(pares)

        refinados = [
            Busca(
                doc_id=item.doc_id,
                titulo=item.titulo,
                texto=item.texto,
                score=float(score),
            )
            for item, score in zip(candidatos, novos_scores)
        ]
        refinados.sort(key=lambda item: item.score, reverse=True)
        return refinados[:top_k]


def imprimir_resultados(titulo: str, itens: List[Busca]) -> None:
    print(f"\n{titulo}")
    print("-" * len(titulo))
    for posicao, item in enumerate(itens, start=1):
        print(f"{posicao:02d}. [{item.doc_id}] {item.titulo} | score={item.score:.4f}")
        print(f"    {item.texto}")


def executar_pipeline(query_usuario: str) -> None:
    pipeline = RagLaboratorio()
    pipeline.construir_indice()

    doc_hipotetico = pipeline.gerar_documento_hipotetico(query_usuario)
    print("QUERY ORIGINAL")
    print(query_usuario)
    print("\nDOCUMENTO HIPOTETICO (HyDE)")
    print(doc_hipotetico)

    top_10 = pipeline.recuperar(query_usuario, top_k=10)
    imprimir_resultados("TOP-10 RECUPERADOS PELO BI-ENCODER + HNSW", top_10)

    top_3 = pipeline.reranquear(query_usuario, top_10, top_k=3)
    imprimir_resultados("TOP-3 FINAIS APOS O CROSS-ENCODER", top_3)


def main() -> None:
    carregar_variaveis_locais()
    executar_pipeline(DEFAULT_QUERY)


if __name__ == "__main__":
    main()
