from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Inicializa o aplicativo FastAPI para o Exercício 2
app = FastAPI(
    title="API de Tradução de Inglês para Francês",
    description="Uma API utilizando o modelo Helsinki-NLP/opus-mt-en-fr da HuggingFace.",
    version="1.0"
)

# Inicializa o pipeline de tradução (o download ocorrerá na primeira execução)
try:
    print("A carregar o modelo de tradução Helsinki-NLP...")
    tradutor = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    print("Modelo de tradução carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    tradutor = None

# Estrutura de dados esperada na requisição HTTP
class RequisicaoTraducao(BaseModel):
    texto_ingles: str

# Endpoint POST para realizar a tradução
@app.post("/traduzir/")
async def traduzir_texto(dados: RequisicaoTraducao):
    if tradutor is None:
        raise HTTPException(status_code=500, detail="O modelo de tradução não foi carregado corretamente.")
    
    if not dados.texto_ingles.strip():
        raise HTTPException(status_code=400, detail="O texto de entrada em inglês não pode estar vazio.")
    
    try:
        # Executa a tradução utilizando o modelo da Hugging Face
        resultado = tradutor(dados.texto_ingles)
        
        return {
            "texto_original_ingles": dados.texto_ingles,
            "texto_traduzido_frances": resultado[0]["translation_text"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a tradução: {str(e)}")

# Endpoint simples para verificar se a API está online
@app.get("/")
def home():
    return {"status": "A API de tradução está a funcionar perfeitamente!"}