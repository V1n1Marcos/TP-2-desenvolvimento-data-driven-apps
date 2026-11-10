from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Inicializa o aplicativo FastAPI
app = FastAPI(
    title="API de Geração de Texto com GPT-2",
    description="Uma API simples que utiliza o modelo GPT-2 da HuggingFace para completar textos.",
    version="1.0"
)

# Inicializa o pipeline de geração de texto da HuggingFace
# O download do modelo será feito automaticamente na primeira execução
try:
    gerador = pipeline("text-generation", model="gpt2")
    print("Modelo carregado com sucesso!")
except Exception as e:
    gerador = None

# Define a estrutura de dados esperada na requisição HTTP usando Pydantic
class RequisicaoTexto(BaseModel):
    texto_entrada: str
    max_new_tokens: int = 50  

# Cria o endpoint POST que receberá os dados
@app.post("/gerar/")
async def gerar_texto(dados: RequisicaoTexto):
    if gerador is None:
        raise HTTPException(status_code=500, detail="O modelo GPT-2 não foi carregado corretamente.")
    
    # Validação simples para evitar requisições sem conteúdo
    if not dados.texto_entrada.strip():
        raise HTTPException(status_code=400, detail="O texto de entrada não pode estar vazio.")
    
    try:
        resultado = gerador(
            dados.texto_entrada, 
            max_new_tokens=dados.max_new_tokens, 
            num_return_sequences=1,
            pad_token_id=50256 # Garante que o GPT-2 lide com o padding corretamente
        )
        
        # Retorna o resultado estruturado em JSON conforme pedido na tarefa
        return {
            "entrada": dados.texto_entrada,
            "texto_gerado": resultado[0]["generated_text"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o modelo: {str(e)}")

# Endpoint opcional apenas para verificar se a API está online
@app.get("/")
def home():
    return {"status": "A API está funcionando perfeitamente!"}


