from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
from ray_vllm_inference.prompt_format import Message

class GenerateRequest(BaseModel):
    """Generate completion request.

        prompt: Prompt to use for the generation
        messages: List of messages to use for the generation
        stream: Bool flag whether to stream the output or not
        max_tokens: Maximum number of tokens to generate per output sequence.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
    
        Note that vLLM supports many more sampling parameters that are ignored here.
        See: vllm/sampling_params.py in the vLLM repository.
        """
    prompt: Optional[str] = None
    messages: Optional[List[Message]] = None
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    ignore_eos: Optional[bool] = False
    
    @validator('prompt', 'messages')
    def validate_prompt_or_messages(cls, v, values):
        # Make sure either prompt or messages is provided
        if 'prompt' in values and not values['prompt'] and 'messages' in values and not values['messages']:
            if v is None:
                raise ValueError("Either 'prompt' or 'messages' must be provided")
        return v
    
    class Config:
        # Allow extra fields to be more tolerant of client implementations
        extra = "ignore"

class GenerateResponse(BaseModel):
    """Generate completion response.

        output: Model output
        prompt_tokens: Number of tokens in the prompt
        output_tokens: Number of generated tokens
        finish_reason: Reason the genertion has finished
    """
    output: str
    prompt_tokens: int
    output_tokens: int
    finish_reason: Optional[str]