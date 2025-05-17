import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np

class InsuranceChatbot:
    def __init__(self):
        self.qdrant = QdrantClient("http://localhost:6333")
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer('intfloat/e5-large-v2')
        print("Loading LLaMA model...")
        self.model_path = "C:/Developer/Models/Llama-3.2-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.bos_token = '<s>'
        self.tokenizer.eos_token = '</s>'
        self.model.resize_token_embeddings(len(self.tokenizer))
        print("All models loaded successfully!")

    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        query_embedding = self.embed_model.encode(query, normalize_embeddings=True)
        print("\nSearching documents...")
        search_result = self.qdrant.search(
            collection_name="insurance_docs",
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=0.7,
            with_payload=True
        )
        if not search_result:
            print("No relevant documents found!")
            return ""
        contexts = []
        for i, result in enumerate(search_result, 1):
            metadata = result.payload.get('metadata', {})
            text = result.payload.get('text', '')
            score = result.score
            file_path = result.payload.get('file_path', 'Unknown')
            preview_text = text[:2000] + ("..." if len(text) > 2000 else "")
            print(f"\nMatch {i}:")
            print(f"File: {os.path.basename(file_path)}")
            print(f"Score: {score:.3f}")
            print(f"SFMA ID: {metadata.get('sfma_id', 'Unknown')}")
            context = f"""
            Source: {os.path.basename(file_path)}
            Relevance: {score:.2f}
            ID: {metadata.get('sfma_id', 'Unknown')}

            Content:
            {preview_text}
            """
            contexts.append(context)
        return "\n" + "=" * 30 + "\n".join(contexts)

    def generate_response(self, query: str, context: str) -> str:
        prompt = f"""You are an insurance regulation expert assistant. Answer the following question using only the provided insurance documents. Be specific and detailed. If you can't find exact information, say so clearly.

    Guidelines:
    - Use bullet points for lists
    - Compare document versions if available
    - Quote specific sections when relevant
    - State clearly if information is incomplete

    Question: {query}

    Context from Insurance Documents:
    {context}

    Detailed Answer:"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    def chat(self):
        print("\nInsurance Expert Assistant ready! (Type 'exit' to end)")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            if user_input:
                self.last_query = user_input
                context = self.get_relevant_context(user_input, top_k=3)
                if not context:
                    print("AI: No relevant information found. Please rephrase your question.")
                    continue
                response = self.generate_response(user_input, context)
                print(f"AI: {response}")

def main():
    try:
        chatbot = InsuranceChatbot()
        chatbot.chat()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure Qdrant is running at http://localhost:6333")

if __name__ == "__main__":
    main()
