# Laboratório 09: Arquitetura RAG Avançada

Pipeline em Python para demonstrar um fluxo de RAG com:

- indexação vetorial em HNSW;
- transformação de query com HyDE;
- recuperação rápida com bi-encoder;
- re-ranking com cross-encoder.

## Estrutura

```text
lab09-rag-avancado/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── data.py
    └── rag_pipeline.py
```

## Como executar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/rag_pipeline.py
```

## O que o código faz

1. Monta uma base simulada com 24 fragmentos de manuais médicos.
2. Gera embeddings densos com `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. Cria um índice `FAISS IndexHNSWFlat` com métrica de produto interno sobre vetores normalizados.
4. Recebe uma query coloquial do usuário.
5. Usa HyDE para gerar um documento hipotético técnico.
6. Recupera os 10 documentos mais próximos no índice HNSW.
7. Reordena os 10 candidatos com `BAAI/bge-reranker-v2-m3`.
8. Exibe os 3 melhores documentos finais.

## Sobre o HNSW

No HNSW, o hiperparâmetro `M` controla quantas conexões cada nó tende a manter no grafo. Quanto maior o `M`, maior o consumo de RAM, porque o índice precisa armazenar mais arestas por vetor. Em troca, a navegação no grafo costuma ficar mais precisa.

O `ef_construction` define o esforço durante a construção do índice. Valores maiores aumentam o tempo de indexação e também elevam o uso de memória temporária, porque mais candidatos precisam ser mantidos durante a montagem da estrutura. Em geral, isso melhora a qualidade das vizinhanças do grafo.

Comparando com um KNN exato, a busca exata não precisa armazenar a malha de vizinhança do HNSW, mas paga caro em tempo de consulta, porque compara a query contra todos os vetores. O HNSW consome mais memória estrutural para reduzir bastante a latência de busca.

## Observação sobre o HyDE

O código tenta usar a API da OpenAI se a variável `OPENAI_API_KEY` estiver definida. Se não estiver, entra em um modo de contingência com reescrita técnica simples para que o laboratório continue executável no ambiente local.

## Exemplo de query

```text
dor de cabeça latejante e luz incomodando
```

## Nota obrigatória

Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por [Seu Nome].
