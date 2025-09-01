import dspy
from openai import OpenAI
import pdb
from langchain_openai import ChatOpenAI
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
import os
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
    RunInfo,
)
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast, List,
)
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from langchain_ibm import ChatWatsonx
from langchain_aws import ChatBedrock
from pydantic import SecretStr

from src.utils import config


class DeepSeekR1ChatOpenAI(ChatOpenAI):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = OpenAI(
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key")
        )

    async def ainvoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append({"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append({"role": "assistant", "content": input_.content})
            else:
                message_history.append({"role": "user", "content": input_.content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)

    def invoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append({"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append({"role": "assistant", "content": input_.content})
            else:
                message_history.append({"role": "user", "content": input_.content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)


class DeepSeekR1ChatOllama(ChatOllama):

    async def ainvoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        org_ai_message = await super().ainvoke(input=input)
        org_content = org_ai_message.content
        reasoning_content = org_content.split("</think>")[0].replace("<think>", "")
        content = org_content.split("</think>")[1]
        if "**JSON Response:**" in content:
            content = content.split("**JSON Response:**")[-1]
        return AIMessage(content=content, reasoning_content=reasoning_content)

    def invoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        org_ai_message = super().invoke(input=input)
        org_content = org_ai_message.content
        reasoning_content = org_content.split("</think>")[0].replace("<think>", "")
        content = org_content.split("</think>")[1]
        if "**JSON Response:**" in content:
            content = content.split("**JSON Response:**")[-1]
        return AIMessage(content=content, reasoning_content=reasoning_content)


def configure_dspy_lm(provider: str, llm_instance: BaseLanguageModel, **kwargs):
    """Mengonfigurasi LLM yang akan digunakan oleh DSPy menggunakan dspy.LM generik."""
    lm = None
    
    # Kondisi untuk provider yang kompatibel dengan API OpenAI
    if provider in ["openai", "azure_openai", "mistral", "alibaba", "moonshot", "unbound", "siliconflow", "modelscope", "grok", "deepseek"]:
        lm = dspy.LM(
            model=kwargs.get("model_name"),
            api_key=kwargs.get("api_key"),
            api_base=kwargs.get("base_url"),
            max_tokens=4000
        )
    # Kondisi untuk provider spesifik yang mungkin memerlukan kelasnya sendiri
    elif provider == "anthropic":
        lm = dspy.Anthropic(
            model=kwargs.get("model_name"),
            api_key=kwargs.get("api_key"),
            api_base=kwargs.get("base_url")
        )
    elif provider == "google":
        lm = dspy.Google(
            model=kwargs.get("model_name"),
            api_key=kwargs.get("api_key")
        )
    elif provider == "ollama":
        # Perbaikan kunci sesuai dokumentasi resmi
        lm = dspy.LM(
            model=f"ollama_chat/{kwargs.get('model_name')}",
            api_base=kwargs.get("base_url", "http://localhost:11434"),
            api_key="" # Ollama lokal tidak memerlukan kunci API
        )
    
    if lm:
        dspy.configure(lm=lm)
        print(f"✅ DSPy configured to use {provider} with model {kwargs.get('model_name')}")
    else:
        print(f"⚠️ DSPy configuration skipped for unsupported provider: {provider}")


def get_llm_model(provider: str, **kwargs):
    """
    Get LLM model
    :param provider: LLM provider
    :param kwargs:
    :return:
    """
    if provider not in ["ollama", "bedrock"]:
        env_var = f"{provider.upper()}_API_KEY"
        api_key = kwargs.get("api_key", "") or os.getenv(env_var, "")
        if not api_key:
            provider_display = config.PROVIDER_DISPLAY_NAMES.get(provider, provider.upper())
            error_msg = f"💥 {provider_display} API key not found! 🔑 Please set the `{env_var}` environment variable or provide it in the UI."
            raise ValueError(error_msg)
        kwargs["api_key"] = api_key

    llm = None  # Inisialisasi llm

    if provider == "anthropic":
        base_url = kwargs.get("base_url") or "https://api.anthropic.com"
        llm = ChatAnthropic(
            model=kwargs.get("model_name", "claude-3-5-sonnet-20241022"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=kwargs.get("api_key"),
        )
    elif provider == 'mistral':
        base_url = kwargs.get("base_url") or os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1")
        api_key = kwargs.get("api_key") or os.getenv("MISTRAL_API_KEY", "")
        llm = ChatMistralAI(
            model=kwargs.get("model_name", "mistral-large-latest"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    # ... (blok provider lainnya mengikuti pola yang sama)
    elif provider in ["openai", "grok", "deepseek", "alibaba", "moonshot", "unbound", "siliconflow", "modelscope"]:
        default_endpoints = {
            "openai": "https://api.openai.com/v1",
            "grok": "https://api.x.ai/v1",
            "deepseek": "https://api.deepseek.com",
            "alibaba": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "moonshot": "https://api.moonshot.cn/v1",
            "unbound": "https://api.getunbound.ai",
            "siliconflow": "https://api.siliconflow.cn/v1/",
            "modelscope": "https://api-inference.modelscope.cn/v1"
        }
        base_url = kwargs.get("base_url") or os.getenv(f"{provider.upper()}_ENDPOINT", default_endpoints.get(provider))
        
        if provider == "deepseek" and kwargs.get("model_name") == "deepseek-reasoner":
             llm = DeepSeekR1ChatOpenAI(
                model=kwargs.get("model_name"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=kwargs.get("api_key"),
            )
        else:
             llm = ChatOpenAI(
                model=kwargs.get("model_name"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=kwargs.get("api_key"),
            )
    elif provider == "google":
        llm = ChatGoogleGenerativeAI(
            model=kwargs.get("model_name", "gemini-2.0-flash-exp"),
            temperature=kwargs.get("temperature", 0.0),
            api_key=kwargs.get("api_key"),
        )
    elif provider == "ollama":
        base_url = kwargs.get("base_url") or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        if "deepseek-r1" in kwargs.get("model_name", "qwen2.5:7b"):
            llm = DeepSeekR1ChatOllama(
                model=kwargs.get("model_name", "deepseek-r1:14b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                base_url=base_url,
            )
        else:
            llm = ChatOllama(
                model=kwargs.get("model_name", "qwen2.5:7b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                num_predict=kwargs.get("num_predict", 1024),
                base_url=base_url,
            )
    elif provider == "azure_openai":
        base_url = kwargs.get("base_url") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_version = kwargs.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        llm = AzureChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            api_version=api_version,
            azure_endpoint=base_url,
            api_key=kwargs.get("api_key"),
        )
    elif provider == "ibm":
        parameters = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("num_ctx", 32000)
        }
        base_url = kwargs.get("base_url") or os.getenv("IBM_ENDPOINT", "https://us-south.ml.cloud.ibm.com")
        llm = ChatWatsonx(
            model_id=kwargs.get("model_name", "ibm/granite-vision-3.1-2b-preview"),
            url=base_url,
            project_id=os.getenv("IBM_PROJECT_ID"),
            apikey=os.getenv("IBM_API_KEY"),
            params=parameters
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Konfigurasi DSPy dipanggil di akhir setelah llm diinisialisasi
    if llm:
        configure_dspy_lm(provider, llm, **kwargs)
        return llm
    else:
        raise ValueError(f"Failed to initialize LLM for provider: {provider}")