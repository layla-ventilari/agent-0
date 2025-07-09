import litellm
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("HF_TEST")
if not api_key:
    raise Exception("HF_TEST variável de ambiente não encontrada")

class Agent:
    def __init__(self):
        self.conversation = []

    def run(self):
        print("Running agent...")
        
        while True:
            user_input = input("You: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the agent.")
                break
            
            user_msg = {"role": "user", "content": user_input}
            self.conversation.append(user_msg)

            response = litellm.completion(
                provider="huggingface",
                model="HuggingFaceH4/zephyr-7b-beta",
                api_key=api_key,
                messages=self.conversation
            )

            print("AI:", response)
            break

if __name__ == "__main__":
    agent = Agent()
    agent.run()
    print("Agent has been initialized and run method called.")
