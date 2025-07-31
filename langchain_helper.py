from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    prompt_template = PromptTemplate(
        input_variables=['animal_type','pet_color'], 
        template="I have a {animal_type} and I want a cool name for it. It is {pet_color}. Suggest 5 cool names."
    )

    name_chain = prompt_template | llm 
    response = name_chain.invoke({"animal_type": animal_type, "pet_color": pet_color})
    return response

if __name__ == "__main__":
    print(generate_pet_name("lizard","white"))