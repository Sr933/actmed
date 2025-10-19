import openai
import numpy as np
import time
import os
import random
OPENAI_API_TYPE = "azure"
OPENAI_API_VERSION = "2024-02-15-preview"


# Set the OpenAI API configurations for different models as environment variables
OPEN_API_BASE_GPT4o_mini    = "https://vdslabazuremloai-fc.openai.azure.com/"
OPEN_API_KEY_GPT4o_mini     = "N1OWS0nAknrQ0s47x0JCiXn8YHb3f6pMM5MbZzRpFH6YJ5n5UuocJQQJ99BEAC5T7U2XJ3w3AAABACOGkauV"
OPEN_API_ENGINE_GPT4o_mini  = "gpt-4o-mini-silas"

OPEN_API_ENGINE_GPT4o= 'gpt-4o-silas'
OPEN_API_BASE_GPT4o = 'https://vdslabazuremloai-fc.openai.azure.com/'
OPEN_API_KEY_GPT4o  = 'N1OWS0nAknrQ0s47x0JCiXn8YHb3f6pMM5MbZzRpFH6YJ5n5UuocJQQJ99BEAC5T7U2XJ3w3AAABACOGkauV'

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
    elif model_name == 'gpt-4.1-mini':
        model = OPEN_API_ENGINE_GPT4_1_mini
        api_key = OPEN_API_KEY_GPT4_1_mini
        api_base = OPEN_API_BASE_GPT4_1_mini
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

def llama_chat(user_prompt, 
               model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
               temperature=1, top_p=0.95):
    """
    Send a chat prompt to a locally hosted model via vLLM.
    """
    client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # Not used but required
    )
    seed=random.randint(0, 10000)
    #Add only return a number to the user prompt at the start before passing to the model
    user_prompt = "Return a number only a number with no reasoning!!! " + user_prompt
    
    user_prompt = user_prompt.strip()
    for _ in range(MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                top_p=top_p,
                n=1
            )
            content = response.choices[0].message.content
            if content:
                return content
        except Exception as e:
            print(f"Retrying due to: {e}")
            time.sleep(random.uniform(0.5, 1.5))

    return "Error: No valid response received."





if __name__ == "__main__":
    # Example usage
    user_prompt = "Name 3 risk factors for stroke?"
    model_name = "gpt-4o"
    response = chat_gpt(user_prompt, model_name)
    print(response)