import os
import json
from typing import List, Annotated
from typing_extensions import TypedDict
import operator

# Importing various LangChain and LangGraph components for RAG (Retrieval Augmented Generation) workflow
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Configure environment variables for tracing and tokenizer behavior
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agent-rag"

# Initialize Language Models
# Primary LLM for general text generation
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Specialized LLM configured for JSON mode responses
llm_json_mode = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0, 
    response_format={"type": "json_object"}
)

# Set up document retrieval and vector store
# List of URLs to be used as source documents
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents from the specified URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents into smaller chunks for better embedding and retrieval
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs_list)

# Create a Chroma vector store with OpenAI embeddings
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding = OpenAIEmbeddings(model="text-embedding-3-large"),
)

# Configure retriever to return top 3 most relevant documents
retriever = vectorstore.as_retriever(k=3)

# Initialize web search tool for additional information retrieval
web_search_tool = TavilySearchResults(k=3)

# Detailed prompts for different stages of the RAG workflow
# Each prompt provides specific instructions for the language model

# Router instructions help decide whether to use vector store or web search
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
                                    
Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

# Instructions for grading the relevance of a document to a user's question
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# Prompt template for grading a single document's relevance
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

# Prompt template for generating an answer based on retrieved context
rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""

# Instructions for grading potential hallucinations in the generated answer
hallucination_grader_instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Prompt template for grading the generated answer against the facts
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

# Instructions for grading the quality of the student's answer
answer_grader_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Prompt template for grading the answer's relevance to the question
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

# Define GraphState using TypedDict for type annotations
class GraphState(TypedDict):
    """
    Represents the state of the RAG workflow.
    
    Tracks various pieces of information as the question is processed:
    - question: The original user query
    - generation: The generated answer
    - web_search: Flag indicating if web search is needed
    - max_retries: Maximum number of generation attempts
    - loop_step: Current iteration of the workflow
    - documents: Retrieved and processed documents
    - final_answer: Final formatted answer
    - structured_output: Structured JSON representation of the result
    """
    question: str
    generation: str
    web_search: str
    max_retries: int
    answers: int
    loop_step: Annotated[int, operator.add]
    documents: List[Document]
    final_answer: str
    structured_output: dict

def format_docs(docs):
    """
    Helper function to format retrieved documents into a single text string.
    
    Args:
        docs (List[Document]): List of retrieved documents
    
    Returns:
        str: Concatenated document contents separated by double newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)

# Node functions define individual steps in the RAG workflow

def retrieve(state):
    """
    Retrieve relevant documents from the vector store based on the user's question.
    
    Args:
        state (GraphState): Current state of the workflow
    
    Returns:
        dict: Updated state with retrieved documents
    """
    question = state["question"]
    documents = retriever.invoke(question)
    # Add source information to each document's metadata
    for doc in documents:
        # Preserve the original URL in 'source_url' before setting 'source' to 'vectorstore'
        doc.metadata["source_url"] = doc.metadata.get("source", "vectorstore")
        doc.metadata["source"] = "vectorstore"
    return {"documents": documents}

def generate(state):
    """
    Generate an answer using the language model based on retrieved documents.
    
    Args:
        state (GraphState): Current state of the workflow
    
    Returns:
        dict: Updated state with generated answer and incremented loop step
    """
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    # Format documents into a single string for context
    docs_txt = format_docs(documents)
    # Fill the RAG prompt with the context and question
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    # Invoke the language model to generate an answer
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}

def grade_documents(state):
    """
    Assess the relevance of each retrieved document to the user's question.
    
    Args:
        state (GraphState): Current state of the workflow
    
    Returns:
        dict: Updated state with filtered relevant documents and web search flag
    """
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        # Format the document and question into the grader prompt
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        # Invoke the JSON-mode language model for grading
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
    return {"documents": filtered_docs, "web_search": web_search}

def web_search(state):
    """
    Perform a web search to retrieve additional information if needed.
    
    Args:
        state (GraphState): Current state of the workflow
    
    Returns:
        dict: Updated state with additional documents from web search
    """
    question = state["question"]
    documents = state.get("documents", [])
    try:
        # Invoke the web search tool with the user's query
        docs = web_search_tool.invoke({"query": question})
        for d in docs:
            if isinstance(d, dict):
                content = d.get("content", "")
                url = d.get("url", "Unknown URL")
                # Create a Document object with content and URL metadata
                web_results_doc = Document(
                    page_content=content,
                    metadata={
                        "source": "websearch",
                        "url": url
                    }
                )
                documents.append(web_results_doc)
                # Debugging: Print the metadata of each websearched document
                print(f"Websearch Document Metadata: {web_results_doc.metadata}")
    except Exception as e:
        # If web search fails, append a failure message with URL as "N/A"
        web_results_doc = Document(
            page_content="Web search failed to return results.",
            metadata={"source": "websearch", "url": "N/A"}
        )
        documents.append(web_results_doc)
        # Debugging: Print the metadata of the failure message
        print(f"Websearch Failure Document Metadata: {web_results_doc.metadata}")
    return {"documents": documents}

# Edge functions determine the flow between nodes based on conditions

def route_question(state):
    """
    Decide whether to use vector store or web search based on the user's question.
    
    Args:
        state (GraphState): Current state of the workflow
    
    Returns:
        str: Next node to transition to ('websearch' or 'vectorstore')
    """
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        return "websearch"
    elif source == "vectorstore":
        return "retrieve"

def decide_to_generate(state):
    """
    Determine whether to proceed with answer generation or perform a web search.
    
    Args:
        state (GraphState): Current state of the workflow
    
    Returns:
        str: Next node to transition to ('websearch' or 'generate')
    """
    web_search = state["web_search"]
    if web_search == "Yes":
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Evaluate the generated answer for correctness and relevance.
    
    Args:
        state (GraphState): Current state of the workflow
    
    Returns:
        str: Decision on the next step ('useful', 'not useful', 'max retries', etc.)
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)

    # Format the hallucination grader prompt with facts and student answer
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    # Invoke the JSON-mode language model for grading hallucinations
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    if grade == "yes":
        # If grounded, proceed to grade the answer's relevance
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            return "useful"
        elif state["loop_step"] <= max_retries:
            return "not useful"
        else:
            return "max retries"
    elif state["loop_step"] <= max_retries:
        return "not supported"
    else:
        return "max retries"

def write_report(state):
    """
    Placeholder function to handle report writing.
    
    Args:
        state (GraphState): Current state of the workflow
    
    Returns:
        str: Next node to transition to ('write_report')
    """
    return "write_report"

def write_final_answer(state):
    """
    Write the final answer based on the generated content.
    
    Args:
        state (GraphState): Current state of the workflow
    
    Returns:
        dict: Updated state with the final answer
    """
    generation = state["generation"]
    loop_step = state.get("loop_step", 0)
    max_retries = state.get("max_retries", 3)
    
    if loop_step > max_retries:
        final_answer = """Unfortunately, after multiple attempts, I could not provide a conclusive answer to your question. 
This might be because:
- The relevant information was not found in the available documents
- The web search did not yield useful results
- The generated answers did not meet our quality standards

Please try rephrasing your question or providing additional context."""
    else:
        final_answer = generation.content
    
    return {"final_answer": final_answer}

def create_structured_output(state):
    """
    Create a structured JSON output summarizing the workflow results.
    
    Args:
        state (GraphState): Current state of the workflow
    
    Returns:
        dict: Structured JSON representation of the final output containing:
            - answer_provided: Indicates if an answer was successfully generated
            - rag_documents: List of documents retrieved from the vector store with content and source URL
            - websearch_documents: List of documents retrieved via web search with content and URL
            - final_answer: The final answer generated by the assistant
    """
    final_answer = state["final_answer"]
    documents = state["documents"]
    answer_provided = "yes" if final_answer else "no"

    # Separate documents based on their source and include both content and URL
    websearch_documents = [
        {
            "content": doc.page_content,
            "url": doc.metadata.get("url", "Unknown URL")
        }
        for doc in documents if doc.metadata.get("source") == "websearch"
    ]

    rag_documents = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source_url", "vectorstore")
        }
        for doc in documents if doc.metadata.get("source") == "vectorstore"
    ]

    structured_output = {
        "answer_provided": answer_provided,
        "rag_documents": rag_documents,
        "websearch_documents": websearch_documents,
        "final_answer": final_answer
    }
    return {"structured_output": structured_output}

# Create and compile the workflow graph using StateGraph

# Initialize the StateGraph with the defined GraphState
workflow = StateGraph(GraphState)

# Add nodes representing each step in the workflow
workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("write_final_answer", write_final_answer)
workflow.add_node("create_structured_output", create_structured_output)

# Set the entry point of the workflow based on the routed question
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

# Define transitions between nodes
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")

# Conditional transitions after grading documents
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)

# Conditional transitions after generating an answer
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": "write_final_answer",
        "not useful": "websearch",
        "max retries": "write_final_answer",
    },
)

# Final transitions to structure the output and end the workflow
workflow.add_edge("write_final_answer", "create_structured_output")
workflow.add_edge("create_structured_output", END)

# Compile the workflow graph into an executable form
graph = workflow.compile()


