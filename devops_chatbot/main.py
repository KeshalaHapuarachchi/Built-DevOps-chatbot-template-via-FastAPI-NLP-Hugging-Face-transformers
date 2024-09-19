import subprocess
import asyncio
from fastapi import FastAPI
from transformers import pipeline

# FastAPI instance for the chatbot
app = FastAPI()

# Load pre-trained NLP model for question answering
nlp_model = pipeline('text-generation', model='gpt-3.5-turbo')

# Define available DevOps commands (abstracted)
DEVOPS_COMMANDS = {
    "deploy": "kubectl apply -f deployment.yaml",
    "status": "kubectl get pods",
    "restart": "systemctl restart nginx",
    "logs": "journalctl -u nginx",
    "disk_usage": "df -h"
}

# Async function to run shell commands
async def run_command(command: str):
    proc = await asyncio.create_subprocess_shell(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode == 0:
        return stdout.decode().strip()
    else:
        return f"Error: {stderr.decode().strip()}"

# Function to handle user queries and map to DevOps commands
async def handle_devops_task(task: str):
    # Check if task is related to known commands
    for command, exec_cmd in DEVOPS_COMMANDS.items():
        if command in task.lower():
            return await run_command(exec_cmd)

    return "Command not recognized or not yet implemented."

# Main chatbot response function
async def chatbot_response(user_input: str):
    # Step 1: Check if the input contains a DevOps task
    devops_response = await handle_devops_task(user_input)
    if devops_response:
        return devops_response

    # Step 2: If no devops task is recognized, respond using the NLP model
    nlp_output = nlp_model(user_input, max_length=50, num_return_sequences=1)
    return nlp_output[0]["generated_text"]

# FastAPI endpoint to interact with the chatbot
@app.post("/chat")
async def chat_endpoint(user_input: str):
    response = await chatbot_response(user_input)
    return {"response": response}

# Example of running the app
# In production, use: `uvicorn <module_name>:app --reload`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
