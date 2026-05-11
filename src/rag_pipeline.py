from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from data import DOCUMENTOS_MEDICOS

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


BI_ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"


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
        api_key = os.getenv("OPENAI_API_KEY")

        if api_key and OpenAI is not None:
            self.client = OpenAI(api_key=api_key)

    def gerar(self, query: str) -> str:
        if self.client is not None:
            prompt = (
                "Você é um assistente clínico de indexação semântica. "
                "Transforme a queixa coloquial abaixo em um pequeno parágrafo técnico, "
                "com termos médicos, hipóteses diagnósticas e sinais associados. "
                "Não escreva avisos. Produza apenas o documento hipotético.\n\n"
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
            "dor de cabeça": "cefaleia",
            "latejante": "pulsátil",
            "luz incomodando": "fotofobia",
            "luz incomoda": "fotofobia",
            "enjoo": "náuseas",
            "visão embaçada": "redução da acuidade visual",
            "tontura": "vertigem",
            "coração acelerado": "taquicardia",
            "falta de ar": "dispneia",
            "queimação": "pirose",
            "ardendo ao urinar": "disúria",
            "muita vontade de urinar": "urgência miccional",
        }

        ponte = texto
        for coloquial, tecnico in trocas.items():
            ponte = ponte.replace(coloquial, tecnico)

        return (
            "Documento hipotético técnico: paciente com "
            f"{ponte}, quadro descrito em terminologia médica e correlacionado com "
            "entidades clínicas prováveis, sintomas associados e linguagem de prontuário."
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
        self.doc_vectors = None

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
        self.doc_vectors = embeddings

    def gerar_documento_hipotetico(self, query: str) -> str:
        return self.hyde.gerar(query)

    def recuperar(self, query: str, top_k: int = 10) -> List[Busca]:
        if self.index is None:
            raise RuntimeError("O índice ainda não foi construído.")

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
        consulta_expandida = f"{query}. Reformulação técnica: {self.gerar_documento_hipotetico(query)}"
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


def main() -> None:
    query_usuario = "dor de cabeça latejante e luz incomodando"
    pipeline = RagLaboratorio()
    pipeline.construir_indice()

    doc_hipotetico = pipeline.gerar_documento_hipotetico(query_usuario)
    print("QUERY ORIGINAL")
    print(query_usuario)
    print("\nDOCUMENTO HIPOTÉTICO (HyDE)")
    print(doc_hipotetico)

    top_10 = pipeline.recuperar(query_usuario, top_k=10)
    imprimir_resultados("TOP-10 RECUPERADOS PELO BI-ENCODER + HNSW", top_10)

    top_3 = pipeline.reranquear(query_usuario, top_10, top_k=3)
    imprimir_resultados("TOP-3 FINAIS APÓS O CROSS-ENCODER", top_3)


if __name__ == "__main__":
    main()
