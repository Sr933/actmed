import openai
import numpy as np
import time
import os
import random

# Azure OpenAI API configuration (version may be updated by Azure)
OPENAI_API_TYPE = "azure"
OPENAI_API_VERSION = "2024-02-15-preview"




def _get_env(name: str) -> str:
    """Fetch required environment variables with a clear error if missing."""
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}.\n"
            f"Please export it in your shell or add it to a .env file."
        )
    return val




# Read Azure OpenAI credentials from environment
# Note: hyphens are not valid in environment variable names; use underscores.
# GPT-4o (prefer GPT_4O_*; fall back to AZURE_OPENAI_* if present)
OPEN_API_BASE_GPT4o = os.getenv("GPT4O_ENDPOINT")
OPEN_API_KEY_GPT4o = os.getenv("GPT4O_KEY")
OPEN_API_ENGINE_GPT4o = os.getenv("GPT4O_DEPLOYMENT")

# GPT-4o-mini
OPEN_API_BASE_GPT4o_mini = os.getenv("GPT4Omini_ENDPOINT")
OPEN_API_KEY_GPT4o_mini = os.getenv("GPT4Omini_KEY")
OPEN_API_ENGINE_GPT4o_mini = os.getenv("GPT4Omini_DEPLOYMENT")

MAX_RETRY = 20


def chat_gpt(user_prompt, model_name, temperature=1, top_p=0.95):
    """
    Sends a chat prompt to the selected model and returns the response.
    
    Args:
      user_prompt (str): The user prompt to send.
      model_name (str): The model to use, e.g., 'gpt-3.5-turbo' or 'gpt-4o-mini'.
      system_prompt (str): Optional system prompt for context.
      temperature (float): Sampling temperature.
      top_p (float): Nucleus sampling probability.
      
    Returns:
      str: The response from the model, or an error message if no valid response is received.
    """
    import numpy as np
    import random
    import time
    
    # Select model configuration based on the specified model_name

    if model_name == 'gpt-4o-mini':
        model = OPEN_API_ENGINE_GPT4o_mini
        api_key = OPEN_API_KEY_GPT4o_mini
        api_base = OPEN_API_BASE_GPT4o_mini
        api_version = OPENAI_API_VERSION
    elif model_name == 'gpt-4o':
        model = OPEN_API_ENGINE_GPT4o
        api_key = OPEN_API_KEY_GPT4o
        api_base = OPEN_API_BASE_GPT4o
        api_version = OPENAI_API_VERSION
    else:
        raise ValueError("Model not supported")
    seed=random.randint(0, 10000)
    success = False
    it = 0
    if not user_prompt:
        return "Error: No prompt provided."
    
    
    while not success and it < MAX_RETRY:
        it += 1
        client = openai.AzureOpenAI(
            azure_endpoint=api_base,
            api_key=api_key,
            api_version=api_version
        )
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                n=1,
                seed=seed
            )
            content = response.choices[0].message.content
            if content:
                success = True
                return content
        except Exception as e:
            # Wait a random short duration before retrying
            time.sleep(random.uniform(0.5, 1.5))
            print(e)
    
    return "Error: No valid response received."

def other_chat(user_prompt, model_name, temperature=1, top_p=0.95):
    """
    Placeholder for other chat function.
    """
    pass

if __name__ == "__main__":
    # Example usage
    user_prompt = "Name 3 risk factors for stroke?"
    model_name = "gpt-4o-mini"
    response = chat_gpt(user_prompt, model_name)
    print(response)