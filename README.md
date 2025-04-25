### Run the instruction model on GPU 0
 `RAY_SERVE_HTTP_PORT=8000 RAY_SERVE_HTTP_HOST=0.0.0.0 serve run ray_vllm_inference.vllm_serve:deployment model="Qwen/Qwen2.5-0.5B-Instruct" tensor_parallel_size=1`

### Run the base model on GPU 1
 `CUDA_VISIBLE_DEVICES=1 RAY_SERVE_HTTP_PORT=8001 RAY_SERVE_HTTP_HOST=0.0.0.0 serve run ray_vllm_inference.vllm_serve:deployment_base model="Qwen/Qwen2.5-0.5B" tensor_parallel_size=1`
