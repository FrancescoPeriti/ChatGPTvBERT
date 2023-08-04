import os
import argparse
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

# Argument parser
parser = argparse.ArgumentParser(prog='ChatGPT prompting', add_help=True)
parser.add_argument('-t', '--trainingData',
                    type=str,
                    default='trainingData',
                    help='Training Data')
parser.add_argument('-a', '--api',
                    type=str,
                    default='your_api',
                    help='Your openai api file')
parser.add_argument('-q', '--query',
                    type=str,
                    default=None,
                    help='Query')
parser.add_argument('-m', '--model',
                    type=str,
                    default='gpt-3.5-turbo',
                    help='Model name')
args = parser.parse_args()

os.environ["OPENAI_API_KEY"] = open(args.api).read()

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True

query = args.query

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
    loader = DirectoryLoader(args.trainingData)
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model=args.model, temperature=0), retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),)

def prompt(query):
  result = chain({"question": query})
  return result['answer']

if __name__ == '__main__':
  chat_history = []
  while True:
      if not query:
          query = input("Prompt: ")
      if query in ['quit', 'q', 'exit']:
          exit(0)
      result = chain({"question": query, "chat_history": chat_history})
      print(result['answer'])

      chat_history.append((query, result['answer']))
      query = None
