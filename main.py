from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
import requests

def get_url_text(url: str) -> str:
  try:
    response = requests.get(url)
    if response.status_code == 200:
      return response.text
    else:
      return f"Error: {response.status_code}"
  except requests.exceptions.RequestException as e:
    return f"Error: {e}"


def sum_numbers(data: str) -> str:
  try:
    numeros = list(map(int, data.split()))
    return f"La suma es {sum(numeros)}"
  except ValueError as e:
    return f"Error: {e}"

ollama_model = ChatOllama(model = "llama3:8b")

# langchain_prompt = PromptTemplate(
#   input_variables = ["user_input"],
#   template="""
#   Eres un experto en Python, en toda tu vida solo viste el lenguaje Python, si recibes cualquier otra
#   consulta, simplemente diras que no tienes esa información. Tambien debe dar respuestas cortas si es posible.
#   Es estrictamente necesario que des las respuestas en español.
  
  
#   Aquí está la consulta del usuario: {user_input}. Usa herramientas si es necesario.
#   """
# )

langchain_prompt = PromptTemplate(
  input_variables = ["user_input"],
  template = """
  Responde estrictamente en español. Si el usuario te saluda (como 'hola', 'buenos días', etc.), responde con un saludo simple y no uses herramientas.
  Usa herramientas solo si la entrada incluye números.
  
  
  Aquí está la consulta del usuario: {user_input}.
  """
)

tools = [
    Tool(
      name = "Sumar numeros",
      func = sum_numbers,
      description = "Suma numeros separados por espacios"
    ),
    Tool(
        name = "Obtener texto de una URL",
        func = get_url_text,
        description = "Recupera el contenido de texto desde una URL proporcionada"
    )
]

langchain_chain = initialize_agent(
  tools = tools,
  llm = ollama_model,
  agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  prompt = langchain_prompt
)

def interact_with_agent(user_query):
  greetings = ["hola", "holaa", "buenos días", "buenas tardes", "buenas noches", "qué tal", "hey"]

  if any(x in user_query.lower() for x in greetings):
    return "¡Hola! ¿Cómo estás?"
  
  try:
    response = langchain_chain.invoke(user_query)
    return response
  except Exception as e:
    return f"Hubo un error procesando tu consulta: {str(e)}"

if __name__ == "__main__":
  while True:
    ask = input("Escribe: ")

    if ask.lower() == 'salir':
      break

    response = interact_with_agent(ask)
    print("Agente:", response)