# vLLM TEI Plugin

A plugin that adds Text Embeddings Inference (TEI) endpoints to the vLLM API server, supporting both embedding and reranking functionalities.

## Features

This plugin adds TEI endpoints to the vLLM API server:

- `/tei/embed` - Text embedding endpoint
- `/tei/rerank` - Text reranking endpoint

## Supported Models

### Embedding Models
- **BGE Large Chinese (bge-large-zh-v1.5)**: High-quality Chinese text embeddings

### Reranker Models  
- **BGE Reranker Large (bge-reranker-large)**: Advanced text reranking capabilities

## Quick Start

### 1. Start the Embedding Service

```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Start BGE Large Chinese embedding service
python3 -m vllm_tei_plugin.entrypoints.openai.api_server \
    --model /opt/huggingface/BAAI/bge-large-zh-v1.5 \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --gpu-memory-utilization 0.4 \
    --served-model-name model \
    --port 8000 \
    --host 0.0.0.0
```

### 2. Start the Reranker Service

```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Start BGE Reranker Large service
python3 -m vllm_tei_plugin.entrypoints.openai.api_server \
    --model /opt/huggingface/BAAI/bge-reranker-large \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --gpu-memory-utilization 0.4 \
    --served-model-name model \
    --port 8000 \
    --host 0.0.0.0
```

## API Usage

### Text Embedding

#### Single Text Embedding
```bash
curl -X POST http://localhost:8000/tei/embed \
    -H "Content-Type: application/json" \
    -d '{
        "inputs": "Your text here",
        "normalize": true,
        "truncate": true
    }'
```

#### Multiple Text Embedding
```bash
curl -X POST http://localhost:8000/tei/embed \
    -H "Content-Type: application/json" \
    -d '{
        "inputs": ["Text 1", "Text 2"],
        "normalize": true,
        "truncate": true
    }'
```

#### OpenAI Compatible Embedding
```bash
curl -X POST http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "input": ["Text 1", "Text 2"],
        "model": "model"
    }'
```

### Text Reranking

```bash
curl -X POST http://localhost:8000/tei/rerank \
    -H "Content-Type: application/json" \
    -d '{
        "query": "Your query text",
        "texts": ["Document 1", "Document 2", "Document 3"],
        "raw_scores": false,
        "return_text": false,
        "truncate": true
    }'
```

## Parameters

### Embedding Parameters
- `inputs`: Text or list of texts to embed
- `normalize`: Whether to normalize the embeddings (default: true)
- `truncate`: Whether to truncate long texts (default: true)

### Reranking Parameters
- `query`: The query text for reranking
- `texts`: List of documents to rerank
- `raw_scores`: Return raw scores instead of normalized ones (default: false)
- `return_text`: Include original text in response (default: false)
- `truncate`: Whether to truncate long texts (default: true)

## Service Configuration

### GPU Settings
- `CUDA_VISIBLE_DEVICES`: Specify which GPU to use
- `--gpu-memory-utilization`: GPU memory usage (0.0-1.0)

### Model Settings
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--dtype`: Model precision (float16, float32)
- `--served-model-name`: Name of the served model

### Server Settings
- `--port`: Server port (default: 8000)
- `--host`: Server host (default: 0.0.0.0)

## Examples

### Complete Embedding Example
```bash
# Start service
export CUDA_VISIBLE_DEVICES=0
python3 -m vllm_tei_plugin.entrypoints.openai.api_server \
    --model /opt/huggingface/BAAI/bge-large-zh-v1.5 \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --gpu-memory-utilization 0.4 \
    --served-model-name model \
    --port 8000 \
    --host 0.0.0.0

# Test embedding
curl -X POST http://localhost:8000/tei/embed \
    -H "Content-Type: application/json" \
    -d '{"inputs": "Hello world", "normalize": true, "truncate": true}'
```

### Complete Reranking Example
```bash
# Start service
export CUDA_VISIBLE_DEVICES=0
python3 -m vllm_tei_plugin.entrypoints.openai.api_server \
    --model /opt/huggingface/BAAI/bge-reranker-large \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --gpu-memory-utilization 0.4 \
    --served-model-name model \
    --port 8000 \
    --host 0.0.0.0

# Test reranking
curl -X POST http://localhost:8000/tei/rerank \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What is machine learning?",
        "texts": ["Machine learning is a subset of AI", "Python is a programming language"],
        "raw_scores": false,
        "return_text": false,
        "truncate": true
    }'
```

## Requirements

- Python 3.12
- CUDA-compatible GPU
- vLLM framework
- Hugging Face models (BGE series)

## License

[Add your license information here] 