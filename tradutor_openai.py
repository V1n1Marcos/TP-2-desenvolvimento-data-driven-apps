import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Inicializa o aplicativo FastAPI
app = FastAPI(
    title="API de Tradução com LangChain e OpenAI",
    description="Protótipo de tradução de inglês para francês utilizando LangChain e modelos da OpenAI.",
    version="1.0"
)

# Configuração da chave de API da OpenAI (certificar de definir a variável de ambiente OPENAI_API_KEY)
# os.environ["OPENAI_API_KEY"] = "sua-chave-api-aqui"

try:
    # Inicializa o modelo de chat da OpenAI via LangChain
    # Utiliza o gpt-4o-mini ou gpt-3.5-turbo para garantir alta velocidade e baixa latência
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Cria o template de prompt estruturado para tradução
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Você é um tradutor profissional especializado em traduzir textos do inglês para o francês com alta precisão e naturalidade."),
        ("human", "Traduza o seguinte texto do inglês para o francês:\n\n{texto_ingles}")
    ])

    # Configura o parser de saída para retornar diretamente a string do texto traduzido
    output_parser = StrOutputParser()

    # Constrói a cadeia (Chain) utilizando a sintaxe moderna de LCEL (LangChain Expression Language)
    traducao_chain = prompt_template | llm | output_parser

except Exception as e:
    print(f"Erro ao configurar a cadeia do LangChain com a OpenAI: {e}")
    traducao_chain = None

# Estrutura de dados esperada na requisição HTTP POST
class RequisicaoTraducaoOpenAI(BaseModel):
    texto_ingles: str

# Endpoint POST para realizar a tradução via OpenAI
@app.post("/traduzir-openai/")
async def traduzir_com_openai(dados: RequisicaoTraducaoOpenAI):
    if traducao_chain is None:
        raise HTTPException(status_code=500, detail="A cadeia de tradução não foi inicializada corretamente. Verifique sua OPENAI_API_KEY.")
    
    if not dados.texto_ingles.strip():
        raise HTTPException(status_code=400, detail="O texto de entrada em inglês não pode estar vazio.")
    
    try:
        # Executa a cadeia passando o input do usuário de forma assíncrona/rápida
        texto_traduzido = await traducao_chain.ainvoke({"texto_ingles": dados.texto_ingles})
        
        return {
            "texto_original_ingles": dados.texto_ingles,
            "texto_traduzido_frances": texto_traduzido.strip(),
            "modelo_utilizado": "gpt-4o-mini"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a requisição na API da OpenAI: {str(e)}")

# Endpoint raiz para verificação
@app.get("/")
def home():
    return {"status": "A API de tradução com LangChain e OpenAI está online!"}