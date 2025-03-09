import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import pandas as pd
import base64
from openai import OpenAI
import re
import json
import uuid
import traceback
from typing import TypedDict, List, Dict, Any, Optional

# Bibliotecas do LangChain para processamento de texto
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langgraph.graph import StateGraph, END


# Função que normaliza o nome de um arquivo, removendo espaços extras
def normalizar_nome_arquivo(nome):
    return re.sub(r'\s+', ' ', nome).strip()


# Função para anonimizar dados sensíveis dentro de um texto,
# como CPF, RG e e-mails, substituindo por marcadores genéricos.
def anonimizar_dados(texto):
    # Anonimiza CPF no formato 123.456.789-00
    texto = re.sub(r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',
                   '[CPF ANONIMIZADO]', texto)
    # Anonimiza RG (pode variar em quantidade de dígitos, mas segue um padrão básico)
    texto = re.sub(r'\b\d{1,2}\.?\d{3}\.?\d{3}-?\d{1}\b',
                   '[RG ANONIMIZADO]', texto)
    # Anonimiza endereços de e-mail
    texto = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL ANONIMIZADO]', texto)
    return texto


# Tupla com as extensões de arquivo permitidas para upload
EXTENSOES_PERMITIDAS = (
    '.pdf', '.docx', '.pptx', '.xlsx', '.csv',
    '.png', '.jpg', '.jpeg',
    '.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'
)

# Criação de um cliente para chamadas de API da OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# Classe usada para tipar o estado do "Agente" (a aplicação),
# definindo quais campos são esperados. Isso ajuda no controle de estado.
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    vetorstore: Optional[FAISS]
    documents: List[Document]
    file_uploads: List[st.runtime.uploaded_file_manager.UploadedFile]
    memory: ConversationBufferMemory
    using_uploaded: bool
    uploaded_dataframes: Dict[str, Any]


# Função para processar planilhas CSV, dividindo as linhas em chunks (segmentos)
# e criando objetos Document do LangChain. Cada chunk recebe metadados para referência.
def processar_planilha(caminho_arquivo, nome_original, origin='uploaded'):
    try:
        df = pd.read_csv(caminho_arquivo)
        header = df.columns.tolist()
        documentos = []
        chunk_size = 10  # tamanho de cada chunk de linhas

        # Percorre o DataFrame em blocos de 10 linhas
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            content = f"Colunas: {', '.join(header)}\n\nDados:\n{chunk.to_markdown(index=False)}"
            # Anonimiza dados sensíveis de cada trecho
            content = anonimizar_dados(content)
            # Cria um Document do LangChain
            documentos.append(Document(
                page_content=content,
                metadata={
                    'source': nome_original,
                    'tipo': 'planilha',
                    'origin': origin,
                    'linha_inicial': i,
                    'linha_final': i + chunk_size
                }
            ))
        return documentos
    except Exception as e:
        st.error(f"Erro ao processar a planilha CSV {nome_original}: {str(e)}")
        return []


# Função para processar planilhas no formato XLSX (Excel).
# Ela percorre cada aba (sheet) do Excel e divide os dados em chunks de 10 linhas.
def processar_planilha_xlsx(caminho_arquivo, nome_original, origin='uploaded'):
    try:
        documentos = []
        chunk_size = 10
        with pd.ExcelFile(caminho_arquivo) as xl:
            # Para cada aba (sheet) do arquivo
            for sheet in xl.sheet_names:
                try:
                    df = xl.parse(sheet_name=sheet)

                    # Converte colunas datetime em string para evitar problemas na hora de converter em markdown
                    for col in df.select_dtypes(include=['datetime64']).columns:
                        df[col] = df[col].astype(str)

                    if df.empty:
                        continue

                    header = [str(col) for col in df.columns.tolist()]

                    # Percorre a aba em blocos de 10 linhas
                    for i in range(0, len(df), chunk_size):
                        chunk = df.iloc[i:i+chunk_size]
                        try:
                            markdown_text = chunk.to_markdown(index=False)
                        except Exception as md_error:
                            markdown_text = str(chunk)

                        content = (f"Sheet: {sheet}\n"
                                   f"Colunas: {', '.join(header)}\n\n"
                                   f"Dados:\n{markdown_text}")

                        # Anonimiza dados sensíveis
                        content = anonimizar_dados(content)
                        # Cria Document do LangChain para cada bloco
                        documentos.append(Document(
                            page_content=content,
                            metadata={
                                'source': nome_original,
                                'tipo': 'planilha',
                                'origin': origin,
                                'sheet': sheet,
                                'linha_inicial': i,
                                'linha_final': i + chunk_size
                            }
                        ))
                except Exception as sheet_error:
                    st.error(
                        f"Erro ao processar a sheet '{sheet}': {str(sheet_error)}")
                    continue

        return documentos
    except Exception as e:
        st.error(
            f"Erro ao processar a planilha XLSX {nome_original}: {str(e)}")
        return []


# Função principal para criar o workflow (fluxo de estados) usando o LangGraph.
# Esse workflow gerencia o processamento de arquivos, a recuperação de documentos e a geração de respostas.
def create_workflow():
    # Criação de embeddings usando o modelo da OpenAI para indexar trechos de documentos
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Configuração do Splitter para dividir documentos em pedaços menores (chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # tamanho de cada chunk em caracteres
        chunk_overlap=200,  # sobreposição entre chunks
        separators=["\n\n", "\n", " ", ""]
    )

    # Função interna do workflow que processa arquivos enviados (upload)
    def process_uploaded_files(state: AgentState):
        if not state['file_uploads']:
            return state

        # Garante que existirá um dicionário para armazenar DataFrames
        if 'uploaded_dataframes' not in state:
            state['uploaded_dataframes'] = {}

        documentos = []
        # Para cada arquivo enviado
        for arquivo in state['file_uploads']:
            nome_original = arquivo.name
            extensao = os.path.splitext(nome_original)[1].lower()
            try:
                # Salva o arquivo temporariamente
                with tempfile.NamedTemporaryFile(delete=False, suffix=extensao) as temp_file:
                    temp_file.write(arquivo.getvalue())
                    caminho_arquivo = temp_file.name

                # Verifica a extensão do arquivo para decidir o tipo de processamento
                if extensao in ('.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'):
                    # Se for arquivo de áudio, faz transcrição
                    docs = transcrever_audio(
                        caminho_arquivo, nome_original, 'uploaded')
                elif extensao in ('.png', '.jpg', '.jpeg'):
                    # Se for imagem, faz "processamento" (descrição usando IA)
                    docs = processar_imagem(caminho_arquivo, 'uploaded')
                elif extensao == '.csv':
                    # Se for CSV, lê em DataFrame e armazena, depois processa
                    df = pd.read_csv(caminho_arquivo)
                    state['uploaded_dataframes'][nome_original] = df
                    docs = processar_planilha(
                        caminho_arquivo, nome_original, origin='uploaded')
                elif extensao == '.xlsx':
                    # Se for XLSX, lê todas as sheets, armazena e processa
                    with pd.ExcelFile(caminho_arquivo) as xl:
                        df_dict = {sheet: xl.parse(
                            sheet_name=sheet) for sheet in xl.sheet_names}
                    state['uploaded_dataframes'][nome_original] = df_dict
                    docs = processar_planilha_xlsx(
                        caminho_arquivo, nome_original, origin='uploaded')
                else:
                    # Caso seja PDF, DOCX, PPTX, etc. Carrega usando loaders do LangChain
                    loader = {
                        '.pdf': PyPDFLoader,
                        '.docx': Docx2txtLoader,
                        '.pptx': UnstructuredPowerPointLoader,
                    }.get(extensao)
                    if loader:
                        docs = loader(caminho_arquivo).load()
                        for doc in docs:
                            doc.page_content = anonimizar_dados(
                                doc.page_content)
                            doc.metadata.update({
                                'tipo': 'documento',
                                'source': nome_original,
                                'origin': 'uploaded'
                            })
                    else:
                        docs = []

                documentos.extend(docs)
                # Remove o arquivo temporário
                os.unlink(caminho_arquivo)
            except Exception as e:
                st.error(f"Erro no processamento de {nome_original}: {str(e)}")

        # Se obteve algum documento após o processamento, cria o índice Vetorstore (FAISS)
        if documentos:
            segmentos = text_splitter.split_documents(documentos)
            state['vetorstore'] = FAISS.from_documents(segmentos, embeddings)
            state['documents'] = segmentos.copy()
            state['using_uploaded'] = True

        # Limpa a lista de arquivos para evitar reprocessamento
        state['file_uploads'] = []
        return state

    # Função para decidir o próximo passo no workflow
    # baseado na última mensagem e no contexto (ex.: se há documentos carregados ou não).
    def route_question(state: AgentState):
        if not state.get('messages'):
            return "generate_direct_answer"
        if state['vetorstore'] is None or len(state['documents']) == 0:
            return "generate_direct_answer"
        last_message = state['messages'][-1].get('content', '')

        # Se o usuário solicitou "transcrição completa", envia para rota específica
        if any(keyword in last_message.lower() for keyword in ["transcrição completa", "transcrição do áudio"]):
            return "handle_special_request"
        return "retrieve_documents"

    # Passo do workflow para lidar com solicitações específicas,
    # como "trazer a transcrição completa do áudio".
    def handle_special_request(state: AgentState):
        # Filtra documentos que são do tipo "áudio"
        audio_docs = [doc for doc in state['documents']
                      if doc.metadata.get('tipo') == 'áudio']
        if audio_docs:
            resposta_lines = ["Transcrições completas de áudio:"]
            for doc in audio_docs:
                transcription = doc.page_content.strip()
                resposta_lines.append(
                    f"Fonte: {doc.metadata['source']}\n{transcription}"
                )
            resposta = "\n\n".join(resposta_lines)
        else:
            resposta = "Nenhuma transcrição de áudio encontrada."

        # Adiciona a resposta ao histórico de mensagens
        state['messages'].append({"role": "assistant", "content": resposta})
        return state

    # Passo do workflow para recuperar documentos relevantes baseados na pergunta do usuário.
    def retrieve_documents(state: AgentState):
        try:
            question = state['messages'][-1]['content']
            if state['vetorstore'] is None:
                raise ValueError(
                    "Nenhum documento foi carregado para consulta")

            # Cria um retriever usando MMR (Maximal Marginal Relevance)
            retriever = state['vetorstore'].as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,         # número de documentos a retornar
                    "fetch_k": 200,  # número total de documentos para considerar antes de ranquear
                    "lambda_mult": 0.5,
                }
            )
            docs = retriever.get_relevant_documents(question)
            if not docs:
                st.warning(
                    "⚠️ Nenhum documento relevante encontrado para a consulta")
            return {**state, "documents": docs}
        except Exception as e:
            st.error(f"Erro na recuperação de documentos: {str(e)}")
            return state

    # Passo do workflow para gerar a resposta final, usando o modelo ChatOpenAI selecionado.
    def generate_answer(state: AgentState):
        selected_model = st.session_state.get("selected_model", "gpt-4o")

        # Define o modelo de chat com base na escolha do usuário
        if selected_model == "gpt-4o":
            chat_model = ChatOpenAI(model=selected_model, temperature=0)
        else:
            chat_model = ChatOpenAI(model=selected_model, temperature=1)

        if not state.get('messages'):
            st.error("Faça uma pergunta sobre o arquivo carregado.")
            return state

        question = state['messages'][-1].get('content', '')
        if not question:
            st.error("Pergunta vazia.")
            return state

        resposta = ""
        try:
            if state.get('documents'):
                # Constrói um contexto unindo o conteúdo dos documentos relevantes
                context = "\n\n".join([
                    f"Fonte: {d.metadata.get('source', 'desconhecido')}\nTrecho: {d.page_content[:1000]}"
                    for d in state['documents']
                ])

                # Se houver planilhas, adiciona instruções sobre como lidar com dados tabulares
                planilha_present = any(d.metadata.get(
                    'tipo') == 'planilha' for d in state.get('documents', []))
                instrucoes_planilha = (
                    "Observação: os dados a seguir são de uma planilha. "
                    "Considere os cabeçalhos e a estrutura tabular para responder com precisão."
                ) if planilha_present else ""

                # Monta o prompt final, incluindo contexto e pergunta
                prompt = f"""
Contexto (use APENAS os documentos carregados):
{context}

{instrucoes_planilha}

Pergunta: {question}
Resposta detalhada e precisa utilizando somente o contexto fornecido e indicando as fontes quando possível:
"""
                resposta = chat_model.predict(prompt)
            else:
                # Se não há documentos, gera resposta direta sem contexto
                resposta = chat_model.predict(
                    f"Responda à pergunta de forma clara e direta: {question}"
                )

            # Armazena a resposta no histórico de mensagens e na memória
            state['messages'].append(
                {"role": "assistant", "content": resposta}
            )
            state['memory'].save_context(
                {"input": question},
                {"output": resposta}
            )
        except Exception as e:
            resposta = f"❌ Erro na geração de resposta: {str(e)}"
            st.error(resposta)

        return state

    # Cria o grafo de estados do workflow
    workflow = StateGraph(AgentState)

    # Adiciona os nós (funções) no workflow
    workflow.add_node("process_files", process_uploaded_files)
    workflow.add_node("handle_special_request", handle_special_request)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("generate_answer", generate_answer)

    # Define o ponto de entrada do fluxo
    workflow.set_entry_point("process_files")

    # Cria condições de transição entre os nós
    workflow.add_conditional_edges(
        "process_files",
        route_question,
        {
            "handle_special_request": "handle_special_request",
            "retrieve_documents": "retrieve_documents",
            "generate_direct_answer": "generate_answer"
        }
    )

    workflow.add_edge("retrieve_documents", "generate_answer")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("handle_special_request", END)

    # Compila e retorna o workflow configurado
    return workflow.compile()


# Função que transcreve áudios usando Whisper (API da OpenAI).
# Só transcreve áudios de até 25MB.
def transcrever_audio(caminho_arquivo, nome_original, origin):
    try:
        # Verifica limite de tamanho
        if os.path.getsize(caminho_arquivo) > 25 * 1024 * 1024:
            return []
        # Abre o arquivo e solicita a transcrição via API
        with open(caminho_arquivo, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="pt"
            )
        # Formata o texto transcrito e anonimiza dados sensíveis
        transcript = anonimizar_dados(
            re.sub(
                r'\n{3,}', '\n\n',
                re.sub(
                    r'([.!?])(\s+)', r'\1\n\n',
                    re.sub(r',(\s+)', r',\1', transcript)
                )
            )
        )
        return [Document(
            page_content=transcript,
            metadata={'source': nome_original,
                      'tipo': 'áudio', 'origin': origin}
        )]
    except Exception as e:
        st.error(f"Erro na transcrição de áudio: {str(e)}")
        return []


# Função para "processar" imagem, basicamente chamando o modelo de chat
# para gerar uma descrição detalhada do que está na imagem.
def processar_imagem(caminho_arquivo, origin):
    try:
        # Função auxiliar para transformar a imagem em Base64
        def codificar_imagem(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        selected_model = st.session_state.get("selected_model", "gpt-4o")

        # Monta a requisição para o Chat, enviando a imagem em Base64
        resposta = client.chat.completions.create(
            model=selected_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Descreva esta imagem detalhadamente considerando:"},
                    {"type": "text", "text": "1. Elementos visuais principais\n2. Texto legível\n3. Contexto geral\n4. Detalhes relevantes\n5. Possíveis significados"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{codificar_imagem(caminho_arquivo)}",
                        "detail": "high"
                    }},
                ]
            }],
            max_tokens=4000,
        )

        # Retorna um Document com a descrição da imagem
        return [Document(
            page_content=anonimizar_dados(resposta.choices[0].message.content),
            metadata={'source': os.path.basename(caminho_arquivo),
                      'tipo': 'imagem', 'origin': origin}
        )]
    except Exception as e:
        st.error(f"Erro no processamento de imagem: {str(e)}")
        return []


# Função auxiliar para exibir um relatório profissional de uma planilha,
# incluindo análises estatísticas e gráficos gerados com Plotly.
def exibir_relatorio_planilha(data, filename):
    st.markdown(f"## Relatório Profissional da Planilha: {filename}")

    # Se 'data' for um DataFrame único
    if isinstance(data, pd.DataFrame):
        df = data.copy()

        # Mostra um resumo estatístico das colunas
        st.markdown("### Resumo Estatístico")
        summary = df.describe(include='all').transpose().reset_index()
        summary.columns = ["Variável"] + list(summary.columns[1:])

        # Cria uma tabela com as estatísticas
        fig_summary = go.Figure(data=[go.Table(
            header=dict(
                values=list(summary.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[summary[col] for col in summary.columns],
                fill_color='lavender',
                align='left'
            )
        )])

        st.plotly_chart(fig_summary, use_container_width=True,
                        key=f"summary_{filename}")

        # Gera gráficos de frequência para colunas categóricas
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            st.markdown(f"### Gráfico de Frequência para {col}")
            freq = df[col].value_counts().reset_index()
            freq.columns = [col, 'Frequência']
            fig_bar = px.bar(freq, x=col, y='Frequência',
                             title=f"Frequência de {col}")
            st.plotly_chart(fig_bar, use_container_width=True,
                            key=f"bar_{filename}_{col}")

        # Gera histogramas e gráficos de tendência para colunas numéricas
        num_cols = df.select_dtypes(include='number').columns
        for col in num_cols:
            st.markdown(f"### Histograma para {col}")
            fig_hist = px.histogram(
                df, x=col, nbins=20, title=f"Histograma de {col}")
            st.plotly_chart(fig_hist, use_container_width=True,
                            key=f"hist_{filename}_{col}")

            st.markdown(f"### Gráfico de Tendência para {col}")
            fig_trend = px.line(df.reset_index(), x='index',
                                y=col, title=f"Tendência de {col}")
            st.plotly_chart(fig_trend, use_container_width=True,
                            key=f"trend_{filename}_{col}")

        # Se houver mais de uma coluna numérica, exibe a matriz de correlação
        if len(num_cols) > 1:
            st.markdown("### Matriz de Correlação")
            corr = df[num_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True,
                                 title="Matriz de Correlação")
            st.plotly_chart(fig_corr, use_container_width=True,
                            key=f"corr_{filename}")

    # Se 'data' for um dicionário, assumimos que é o caso de múltiplas sheets (XLSX)
    elif isinstance(data, dict):
        for sheet, df in data.items():
            st.markdown(f"## Relatório para Sheet: {sheet}")
            if not df.empty:
                exibir_relatorio_planilha(df, f"{filename}_{sheet}")
            else:
                st.info("Sheet vazia")


# Função principal da aplicação Streamlit
def main():
    st.title("Assistente Inteligente Multimodal 📚🎤📷🤖")

    # Inicializa o workflow se ainda não existir no estado
    if 'workflow' not in st.session_state:
        st.session_state.workflow = create_workflow()

    # Gera um ID único de sessão
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    session_id = st.session_state.session_id

    # Dicionário que armazena o estado do usuário baseado no session_id
    if 'user_sessions' not in st.session_state:
        st.session_state.user_sessions = {}
    if session_id not in st.session_state.user_sessions:
        initial_state = {
            "messages": [],
            "vetorstore": None,
            "documents": [],
            "file_uploads": [],
            "memory": ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="input",
                output_key="output"
            ),
            "using_uploaded": False,
            "uploaded_dataframes": {}
        }
        st.session_state.user_sessions[session_id] = initial_state

    user_session = st.session_state.user_sessions[session_id]

    # Barra lateral para gerenciamento de conteúdo
    with st.sidebar:
        st.header("📁 Gerenciamento de Conteúdo")
        # Permite escolher o modelo de ChatOpenAI
        modelo_escolhido = st.selectbox("Selecione o modelo para respostas:",
                                        ["gpt-4o", "o1-mini"])
        st.session_state["selected_model"] = modelo_escolhido

        with st.expander("📌 Instruções Rápidas", expanded=True):
            st.markdown("""
            - Formatos suportados: Documentos, Imagens, Áudios e Planilhas  
            - Tamanho máximo para arquivo de áudio: 25MB  
            - Carregar arquivos substitui o conhecimento atual
            """)

        # Upload de arquivos
        arquivos_enviados = st.file_uploader(
            "Carregar arquivos (arraste ou selecione)",
            type=EXTENSOES_PERMITIDAS,
            accept_multiple_files=True
        )

        # Se existir arquivo(s) enviado(s), atualiza o estado e processa
        if arquivos_enviados:
            user_session['file_uploads'] = arquivos_enviados
            if st.button("⏳ Processar Conteúdo", use_container_width=True):
                with st.spinner("Processando arquivos..."):
                    st.session_state.workflow.invoke(user_session)

        # Se não há arquivos novos e já houve upload antes, limpa a base
        elif user_session.get('using_uploaded', False):
            st.info(
                "Arquivos carregados foram fechados. A base de conhecimento está vazia."
            )
            user_session['vetorstore'] = None
            user_session['documents'] = []
            user_session['file_uploads'] = []
            user_session['using_uploaded'] = False
            user_session['uploaded_dataframes'] = {}

    # Exibe as mensagens trocadas até agora (histórico)
    for mensagem in user_session['messages']:
        with st.chat_message(mensagem["role"]):
            st.markdown(mensagem["content"])

    # Área principal do chat, onde o usuário digita a pergunta
    with st.container():
        pergunta = st.chat_input(
            "🎤 Faça sua pergunta sobre documentos, imagens, áudios ou planilhas..."
        )
        if pergunta:
            # Adiciona a pergunta ao histórico
            user_session['messages'].append(
                {"role": "user", "content": pergunta}
            )
            with st.chat_message("user"):
                st.markdown(pergunta)

            # Processa a pergunta usando o workflow
            try:
                with st.spinner("Processando sua pergunta..."):
                    for output in st.session_state.workflow.stream(user_session):
                        for key in output:
                            user_session.update(output[key])

                # Exibe a última resposta gerada
                with st.chat_message("assistant"):
                    st.markdown(user_session['messages'][-1]['content'])

                    # Exibe as referências (fontes) dos últimos documentos usados
                    if 'documents' in user_session and user_session['documents']:
                        with st.expander("📚 Fontes de Referência"):
                            for i, doc in enumerate(user_session['documents'][-5:]):
                                nome = normalizar_nome_arquivo(
                                    doc.metadata['source'])
                                link = doc.metadata.get('link', '')
                                # Se houver link, exibe como link clicável; caso contrário, mostra somente o nome
                                st.markdown(
                                    f"**Documento {i+1}:** {'[📎 ' + nome + '](' + link + ')' if link else f'`{nome}`'}"
                                )
                                st.caption(
                                    f"Tipo: {doc.metadata.get('tipo', 'documento')}\n{doc.page_content[:300]}..."
                                )

            except Exception as e:
                st.error(f"❌ Erro na geração de resposta: {str(e)}")

            # Se a pergunta menciona "relatório" e "planilha", gera o relatório para as planilhas carregadas
            if re.search(r'relat[oó]rio', pergunta, re.IGNORECASE) and re.search(r'planilha', pergunta, re.IGNORECASE):
                if user_session.get('uploaded_dataframes'):
                    st.info(
                        "Gerando relatório profissional da(s) planilha(s) carregada(s)...")
                    for filename, data in user_session['uploaded_dataframes'].items():
                        exibir_relatorio_planilha(data, filename)
                else:
                    st.info("Nenhuma planilha carregada para gerar relatório.")

        # Botão para limpar todo o histórico de conversas
        if st.button("🧹 Limpar Histórico", use_container_width=True):
            user_session['messages'] = []
            user_session['memory'].clear()
            user_session['using_uploaded'] = False
            user_session['uploaded_dataframes'] = {}
            st.rerun()


# Execução do Streamlit
if __name__ == "__main__":
    main()
