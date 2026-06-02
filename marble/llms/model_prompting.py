import litellm
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional
from litellm.types.utils import Message

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

from marble.llms.error_handler import api_calling_error_exponential_backoff

_HF_LOCAL_PREFIX = "local_hf/"
_hf_pipelines: Dict[str, Any] = {}


def _local_hf_prompting(
    model_name: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> List[Message]:
    """Run inference with a locally loaded HuggingFace model."""
    if model_name not in _hf_pipelines:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
        _hf_pipelines[model_name] = hf_pipeline(
            "text-generation", model=hf_model, tokenizer=tokenizer
        )
    pipe = _hf_pipelines[model_name]
    do_sample = temperature > 0
    output = pipe(
        messages,
        max_new_tokens=max_new_tokens or 512,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
    )
    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        content = generated[-1]["content"]
    else:
        content = generated
    return [Message(content=content, role="assistant")]


@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def model_prompting(
    llm_model: str,
    messages: List[Dict[str, str]],
    return_num: Optional[int] = 1,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    mode: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> List[Message]:
    """
    Select model via router in LiteLLM with support for function calling.
    Prefix the model name with 'local_hf/' to use a locally loaded HuggingFace model.
    """
    if llm_model.startswith(_HF_LOCAL_PREFIX):
        return _local_hf_prompting(
            llm_model[len(_HF_LOCAL_PREFIX):],
            messages,
            max_token_num or 512,
            temperature or 0.0,
        )

    # litellm.set_verbose=True
    if "together_ai/TA" in llm_model:
        base_url = "https://api.ohmygpt.com/v1"
    else:
        base_url = None
    completion = litellm.completion(
        model=llm_model,
        messages=messages,
        max_tokens=max_token_num,
        n=return_num,
        top_p=top_p,
        temperature=temperature,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        base_url=base_url,
    )
    message_0: Message = completion.choices[0].message
    assert message_0 is not None
    assert isinstance(message_0, Message)
    return [message_0]
