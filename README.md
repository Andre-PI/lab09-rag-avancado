# Laboratorio 09: Arquitetura RAG Avancada

Pipeline em Python para demonstrar um fluxo de RAG com:

- indexacao vetorial em HNSW;
- transformacao de query com HyDE;
- recuperacao rapida com bi-encoder;
- re-ranking com cross-encoder.

## Estrutura

```text
lab09-rag-avancado/
|-- README.md
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- src/
    |-- data.py
    `-- rag_pipeline.py
```

## Execucao

No terminal:

```bash
python src/rag_pipeline.py
```

Ou, se quiser isolar dependencias:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python src/rag_pipeline.py
```

## Arquivo .env

O projeto aceita uma chave opcional em `.env`.

Arquivo `.env`:

```env
OPENAI_API_KEY=
```

Se a chave estiver preenchida, o HyDE usa a OpenAI.

Se estiver vazia, o codigo entra automaticamente no modo de contingencia local.

O script nao pede nenhuma entrada interativa, o que facilita testes automatizados.

## O que o codigo faz

1. Monta uma base simulada com 24 fragmentos de manuais medicos.
2. Gera embeddings densos com `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. Cria um indice `FAISS IndexHNSWFlat` com metrica de produto interno sobre vetores normalizados.
4. Recebe uma query coloquial do usuario.
5. Usa HyDE para gerar um documento hipotetico tecnico.
6. Recupera os 10 documentos mais proximos no indice HNSW.
7. Reordena os 10 candidatos com `BAAI/bge-reranker-v2-m3`.
8. Exibe os 3 melhores documentos finais.

## Sobre o HNSW

No HNSW, o hiperparametro `M` controla quantas conexoes cada no tende a manter no grafo. Quanto maior o `M`, maior o consumo de RAM, porque o indice precisa armazenar mais arestas por vetor. Em troca, a navegacao no grafo costuma ficar mais precisa.

O `ef_construction` define o esforco durante a construcao do indice. Valores maiores aumentam o tempo de indexacao e tambem elevam o uso de memoria temporaria, porque mais candidatos precisam ser mantidos durante a montagem da estrutura. Em geral, isso melhora a qualidade das vizinhancas do grafo.

Comparando com um KNN exato, a busca exata nao precisa armazenar a malha de vizinhanca do HNSW, mas paga caro em tempo de consulta, porque compara a query contra todos os vetores. O HNSW consome mais memoria estrutural para reduzir bastante a latencia de busca.

## Nota obrigatoria

Partes deste laboratorio foram geradas/complementadas com IA, revisadas e validadas por Andre Lucas Francino Castelo Branco.
- data.py: gerada por IA
- rag_pipeline.py: prompts gerados por IA 