from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Inicializa o aplicativo FastAPI
app = FastAPI(
    title="API de Tradução (Inglês para Alemão) com LangChain e Hugging Face",
    description="Protótipo utilizando o modelo Helsinki-NLP/opus-mt-en-de integrado via LangChain.",
    version="1.0"
)

# Inicializa a integração do modelo Hugging Face no LangChain
try:
    print("A carregar o modelo Helsinki-NLP/opus-mt-en-de...")
    # O pipeline encapsula o modelo da Hugging Face para uso no LangChain
    hf_pipeline = HuggingFacePipeline.from_model_id(
        model_id="Helsinki-NLP/opus-mt-en-de",
        task="translation_en_to_de",
        pipeline_kwargs={"max_length": 512}
    )
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo Hugging Face: {e}")
    hf_pipeline = None

# Cria o template de prompt adequado para modelos de tradução direta
prompt_template = PromptTemplate(
    input_variables=["texto_ingles"],
    template="{texto_ingles}"
)

# Configura o parser de saída
output_parser = StrOutputParser()

# Constrói a cadeia (Chain) utilizando a sintaxe moderna de LCEL (LangChain Expression Language)
if hf_pipeline:
    traducao_de_chain = prompt_template | hf_pipeline | output_parser
else:
    traducao_de_chain = None

# Estrutura de dados esperada na requisição HTTP POST
class RequisicaoTraducaoAlemao(BaseModel):
    texto_ingles: str

# Endpoint POST para realizar a tradução para o alemão
@app.post("/traduzir-alemaum/")
async def traduzir_para_alemao(dados: RequisicaoTraducaoAlemao):
    if traducao_de_chain is None:
        raise HTTPException(status_code=500, detail="A cadeia de tradução não foi inicializada corretamente.")
    
    if not dados.texto_ingles.strip():
        raise HTTPException(status_code=400, detail="O texto de entrada em inglês não pode estar vazio.")
    
    try:
        # Executa a cadeia passando o input do usuário de forma assíncrona
        texto_traduzido = await traducao_de_chain.ainvoke({"texto_ingles": dados.texto_ingles})
        
        return {
            "texto_original_ingles": dados.texto_ingles,
            "texto_traduzido_alemao": texto_traduzido.strip(),
            "modelo_utilizado": "Helsinki-NLP/opus-mt-en-de"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a tradução: {str(e)}")

# Endpoint raiz para verificação
@app.get("/")
def home():
    return {"status": "A API de tradução de inglês para alemão (LangChain + Hugging Face) está online!"}