import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

def text_generator(biased_text):

    #Conects to Gemini API an return a unbiased text

    genai.configure(api_key=os.getenv('API_KEY_GOOGLE'))
    model=genai.GenerativeModel('gemini-2.5-flash')

    prompt = ''' Provide ONLY the rewritten text, without any introductory comments.
    Rewrite the following text to make it completely neutral, inclusive, and free from discriminatory biases. Ensure to:
    Remove sexist language: Use neutral alternatives (e.g., "people" instead of "men", "team" instead of "guys").
    Avoid stereotypes: Do not associate roles, abilities, or behaviors with gender, ethnicity, age, etc.
    Disability inclusivity: Use "people with disabilities" instead of derogatory terms.
    Cultural/religious neutrality: Do not assume all people share the same beliefs or practices.
    Non-binary language: Choose inclusive terms (e.g., "those who" instead of "those that").
    Objective tone: Avoid subjective adjectives or value judgments.'''

    complete_input = f"{prompt}\n\nText to rescribe:\n{biased_text}"
    response = model.generate_content(complete_input)
    debiased_response = response.candidates[0].content.parts[0].text

    return debiased_response
