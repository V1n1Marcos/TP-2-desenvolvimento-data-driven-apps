from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms.fake import FakeListLLM
from langchain_classic.chains import LLMChain  # <-- Atualizado para langchain-classic
from langchain_core.prompts import PromptTemplate

# Inicializa o aplicativo FastAPI
app = FastAPI(
    title="Chatbot Simulado com LangChain e Fake LLM",
    description="Protótipo de chatbot utilizando respostas pré-definidas para testes rápidos.",
    version="1.0"
)

# Define as respostas simuladas que o FakeLLM irá retornar em sequência
respostas_falsas = [
    "Olá! Como posso ajudar você com os seus estudos de Data-Driven Apps?",
    "Essa é uma excelente pergunta sobre LangChain e arquitetura de modelos.",
    "O uso de Fake LLM acelera drasticamente os testes unitários e de integração.",
    "No momento, estou operando em modo de simulação offline."
]

# Configura o FakeListLLM com a lista de respostas sequenciais
fake_llm = FakeListLLM(responses=respostas_falsas)

# Define um template de prompt simples
prompt_template = PromptTemplate(
    input_variables=["pergunta"],
    template="Usuário perguntou: {pergunta}\nAssistente:"
)

# Cria a cadeia (Chain) utilizando o LangChain clássico/comunitário
chatbot_chain = LLMChain(llm=fake_llm, prompt=prompt_template)

# Estrutura de dados para a requisição POST via Pydantic
class RequisicaoChat(BaseModel):
    pergunta: str

# Endpoint POST do chatbot
@app.post("/chat/")
async def interagir_chatbot(dados: RequisicaoChat):
    if not dados.pergunta.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")
    
    try:
        # Executa a cadeia passando o input do usuário
        # O FakeListLLM consome a próxima resposta da lista sequencial configurada
        resposta_gerada = chatbot_chain.run(pergunta=dados.pergunta)
        
        return {
            "pergunta_recebida": dados.pergunta,
            "resposta_simulada": resposta_gerada.strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a requisição: {str(e)}")

# Endpoint raiz para verificação
@app.get("/")
def home():
    return {"status": "O Chatbot com Fake LLM está online!"}