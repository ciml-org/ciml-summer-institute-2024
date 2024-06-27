#!/usr/bin/env python
# coding: utf-8

# Advanced RAG on Hugging Face documentation using LangChain
# taken from https://huggingface.co/learn/cookbook/en/advanced_rag
# and also from https://python.langchain.com/v0.2/docs/integrations/document_loaders/url/

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

#from langchain.vectorstores import FAISS   #Facebook tool
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional, List, Tuple
from langchain.docstore.document import Document as LangchainDocument


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: str, #EMBEDDING_MODEL_NAME
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

# -----------------------------------------------
#   main starts here
# -----------------------------------------------

if __name__ == '__main__':

  #First set up loader and get web pages
  urls = [
    "https://slurm.schedmd.com/quickstart.html",
    "https://slurm.schedmd.com/man_index.html"
  ]

  loader = UnstructuredURLLoader(urls=urls)
  raw_pages = loader.load_and_split()

  #raw_pages is a list
  print('Num of raw pages after split:',len(raw_pages))

  #Second set up a model to split the web pages
  EMBEDDING_MODEL_NAME = "thenlper/gte-small"

  # We use a hierarchical list of separators specifically tailored for splitting Markdown documents
  # This list is taken from LangChain's MarkdownTextSplitter class
  MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
    ]

  docs_processed = split_documents(
    256,        # chunk size 
    raw_pages,  
    tokenizer_name=EMBEDDING_MODEL_NAME,
  )

  print('Length of docs:', len(docs_processed))

  #Third, set up embedding model and create vector database
  embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=False,  #True,   #this might cause some fork issues?
    model_kwargs={"device": "cpu"},   # "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
  )

  KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
  )

  #Now, embed a user query in the same space, show sample document
  user_query = "How to create a slurm job?"
  query_vector = embedding_model.embed_query(user_query)

  print(f"\nStarting retrieval for {user_query=}...")
  retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
  #print(
  #  "\n==================================Top document=================================="
  #)
  #print(retrieved_docs[0].page_content)
  #print("==================================Metadata==================================")
  #print(retrieved_docs[0].metadata)


  #Now setup the hugging face pipeline
  import huggingface_hub
  from transformers import AutoTokenizer
  import transformers
  import torch

  #Set up model and tokenizer
  model="meta-llama/Meta-Llama-3-8B-Instruct"
  tokenizer = AutoTokenizer.from_pretrained(model)
  #print(tokenizer)

  #Set up prompt template with a place for context informatoin
  prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
  give a comprehensive answer to the question.
  Respond only to the question asked, response should be concise and relevant to the question.
  Provide the number of the source document when relevant.
  If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
  {context}
  ---
  Now here is the question you need to answer.

  Question: {question}""",
    },
  ]
  RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    prompt_in_chat_format, tokenize=False, add_generation_prompt=True
  )
 #print(RAG_PROMPT_TEMPLATE)


  #set up actual prompt with context consisting of retreived docs
  retrieved_docs_text = [
    doc.page_content for doc in retrieved_docs
  ]  # We only need the text of the documents
  context = "\nExtracted documents:\n"
  context += "".join(
    [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
  )

  final_prompt = RAG_PROMPT_TEMPLATE.format(
    question=user_query, context=context
  )


  #get device integer, for the pipeline definition below
  #torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
  device2use=0 if torch.cuda.is_available()  else -1  #-1 means default to cpu
  print('MYINFO device to use',device2use, ' currdev', torch.cuda.current_device())

  #set up the function 
  my_pipe2 = transformers.pipeline(
    #"text-generation",
    model=model,
    #for gpu : 
    torch_dtype=torch.float16,
    #torch_dtype=torch.float32,  #for cpu use this
    device_map="auto",
    #device=device2use
  )
  print('pipeline2 defined')
  mem_allocated = torch.cuda.memory_allocated()
  print('MYINFO mem allocated before results:', mem_allocated)

  #now call the function with the prompt as input and other options
  results_list = my_pipe2(
    final_prompt,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=500, #num new tokens to generate
  )

  mem_allocated = torch.cuda.memory_allocated()
  print('MYINFO mem allocated aft results:', mem_allocated)

  for result in results_list:   #result is a python dict object
    print(' ----------------- Generated Text Result --------------------------')
    print(f"Result: {result['generated_text']}")

