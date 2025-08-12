export CUDA_VISIBLE_DEVICES=0

python3 -m vllm_tei_plugin.entrypoints.openai.api_server \
    --model BAAI/bge-large-zh-v1.5 \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --gpu-memory-utilization 0.4 \
    --served-model-name model \
    --port 8000 \
    --host 0.0.0.0 \
