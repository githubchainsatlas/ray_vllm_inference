from typing import Dict, List, AsyncGenerator
import logging
import uuid
import traceback
from http import HTTPStatus
from ray import serve
from ray.serve import Application
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request as FastAPIRequest
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import StreamingResponse, Response, JSONResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from ray_vllm_inference.prompt_format import Message
from ray_vllm_inference.model_config import load_model_config
from ray_vllm_inference.protocol import GenerateRequest, GenerateResponse

logger = logging.getLogger("ray.serve")
# Set to debug level to get more information
logger.setLevel(logging.DEBUG)

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Log more details about the validation error
    logger.error(f"Validation error: {exc}")
    logger.error(f"Request body: {await request.body()}")
    logger.error(f"Error details: {exc.errors()}")
    return JSONResponse(
        status_code=HTTPStatus.BAD_REQUEST.value,
        content={"detail": f"Error parsing JSON payload: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, 
        content={"detail": f"Server error: {str(exc)}"}
    )

def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code.value, content={"detail":message})

@serve.deployment(name='VLLMInference', 
                  num_replicas=1,
                  ray_actor_options={"num_gpus": 2.0})
@serve.ingress(app)
class VLLMGenerateDeployment:
    def __init__(self, **kwargs):
        """
        Construct a VLLM deployment.
        """
        if 'tensor_parallel_size' in kwargs and isinstance(kwargs['tensor_parallel_size'], str):
            kwargs['tensor_parallel_size'] = int(kwargs['tensor_parallel_size'])

        args = AsyncEngineArgs(**kwargs)
        logger.info(f"Initializing with args: {args}")
        self.engine = AsyncLLMEngine.from_engine_args(args)
        engine_model_config = self.engine.engine.get_model_config()
        self.tokenizer = self.engine.engine.tokenizer
        self.max_model_len = kwargs.get('max_model_len', engine_model_config.max_model_len)

        # Log tokenizer type for debugging
        logger.info(f"Tokenizer type: {type(self.tokenizer).__name__}")
        logger.info(f"AsyncLLMEngine type: {type(self.engine).__name__}")
        
        try:
            self.model_config = load_model_config(args.model)
            logger.info(f"Loaded model config for {args.model}")
        except FileNotFoundError:
            logger.warn(f"No model config for: {args.model}")
            self.model_config = None

    def _next_request_id(self):
        return str(uuid.uuid1().hex)

    def _check_length(self, prompt:str, request:GenerateRequest) -> int:
        """Check if the prompt length is within the model's context length."""
        try:
            # First try using the engine's tokenize method
            input_ids = self.engine.engine.tokenize([prompt])[0]
            token_num = len(input_ids)
            logger.debug(f"Tokenization successful, got {token_num} tokens")
        except Exception as e:
            logger.error(f"Error tokenizing with engine: {e}")
            # Fallback: just estimate token count using simple heuristic
            token_num = len(prompt.split())
            logger.warning(f"Using estimated token count: {token_num}")
        
        if request.max_tokens is None:
            request.max_tokens = self.max_model_len - token_num
        
        if token_num + request.max_tokens > self.max_model_len:
            raise ValueError(
            f"This model's maximum context length is {self.max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the prompt, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the prompt or completion.")
        
        return token_num

    async def _stream_results(self, output_generator) -> AsyncGenerator[bytes, None]:
        num_returned = 0
        async for request_output in output_generator:
            output = request_output.outputs[0]
            text_output = output.text[num_returned:]
            response = GenerateResponse(output=text_output, 
                             prompt_tokens=len(request_output.prompt_token_ids), 
                             output_tokens=1, 
                             finish_reason=output.finish_reason)
            yield (response.json() + "\n").encode("utf-8")
            num_returned += len(text_output)

    async def _abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    @app.get("/health")
    async def health(self) -> Response:
        """Health check."""
        return Response(status_code=200)

    @app.post("/generate")
    async def generate(self, request_raw: FastAPIRequest) -> Response:
        """Generate completion for the request."""
        try:
            # Log raw request data for debugging
            body = await request_raw.body()
            logger.debug(f"Raw request body: {body}")
            
            # Try to parse as JSON first to catch any JSON parsing errors explicitly
            try:
                import json
                json_body = json.loads(body)
                logger.debug(f"Parsed JSON: {json_body}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                return create_error_response(
                    HTTPStatus.BAD_REQUEST, 
                    f"Invalid JSON: {str(e)}"
                )
            
            # Now try to validate with Pydantic model
            try:
                request = GenerateRequest.parse_obj(json_body)
                logger.debug(f"Validated request: {request}")
            except Exception as e:
                logger.error(f"Request validation error: {str(e)}")
                return create_error_response(
                    HTTPStatus.BAD_REQUEST, 
                    f"Request validation error: {str(e)}"
                )

            if not request.prompt and not request.messages:
                return create_error_response(HTTPStatus.BAD_REQUEST, "Missing parameter 'prompt' or 'messages'")

            if request.prompt:
                 prompt = request.prompt
            else:
                if self.model_config:
                    prompt = self.model_config.prompt_format.generate_prompt(request.messages)
                else:
                    return create_error_response(HTTPStatus.BAD_REQUEST, 'Parameter "messages" requires a model config')

            logger.debug(f"Processing prompt: {prompt[:50]}...")
            
            try:
                # Check length but we don't need token IDs anymore
                token_num = self._check_length(prompt, request)
                logger.debug(f"Token count: {token_num}")
            except Exception as e:
                logger.error(f"Error checking length: {str(e)}")
                return create_error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"Error checking prompt length: {str(e)}"
                )

            request_dict = request.dict(exclude=set(['prompt', 'messages', 'stream']))
            sampling_params = SamplingParams(**request_dict)
            request_id = self._next_request_id()

            # Use the correct API for AsyncLLMEngine.generate
            # Pass the prompt directly instead of prompt_token_ids
            try:
                logger.debug(f"Calling engine.generate with prompt and parameters")
                output_generator = self.engine.generate(
                    prompt=prompt,  # Use the text prompt directly
                    sampling_params=sampling_params,
                    request_id=request_id
                )
            except Exception as e:
                logger.error(f"Error in engine.generate: {str(e)}")
                logger.error(traceback.format_exc())
                return create_error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"Error generating response: {str(e)}"
                )
            
            if request.stream:
                background_tasks = BackgroundTasks()
                background_tasks.add_task(self._abort_request, request_id)
                return StreamingResponse(self._stream_results(output_generator), 
                                        background=background_tasks)
            else:
                final_output = None
                try:
                    async for request_output in output_generator:
                        if await request_raw.is_disconnected():
                            await self.engine.abort(request_id)
                            return Response(status_code=200)
                        final_output = request_output
                except Exception as e:
                    logger.error(f"Error in processing generator: {str(e)}")
                    logger.error(traceback.format_exc())
                    return create_error_response(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        f"Error processing response: {str(e)}"
                    )

                if final_output is None:
                    return create_error_response(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "No output generated"
                    )

                text_outputs = final_output.outputs[0].text
                prompt_tokens = len(final_output.prompt_token_ids)
                output_tokens = len(final_output.outputs[0].token_ids)
                finish_reason = final_output.outputs[0].finish_reason
                return GenerateResponse(output=text_outputs, prompt_tokens=prompt_tokens, 
                                        output_tokens=output_tokens, finish_reason=finish_reason)

        except ValueError as e:
            logger.error(f"Value error: {str(e)}")
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:
            logger.error('Error in generate()', exc_info=1)
            raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, f'Server error: {str(e)}')

def deployment(args: Dict[str, str]) -> Application:
    return VLLMGenerateDeployment.bind(**args)