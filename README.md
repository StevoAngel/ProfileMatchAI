# ProfileMatchAI

Tool designed to upload a set of résumés and a job vacancy description.  
The system identifies and returns candidates whose profiles best match the job requirements.

**Important prerequisite:** 
This project uses Ollama to run a local _Large Language Model_ (`mistral`) , then it is necessary to install Ollama in your system before running the app.
Download Link: https://ollama.com/

## Features
- Upload multiple CVs (PDF only)
- Intelligent profile matching using NLP and LLMs: (<ins>In progress...</ins>)
- Export top matches (<ins>In progress...</ins>)

## Getting Started
### Option 1: Run locally:

1. Clone this repository: https://github.com/StevoAngel/ProfileMatchAI
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

### Option 2: Run with Docker (**recommended**):
This project includes a `Dockerfile` and a batch file (`init.bat`) for an easy execution on Windows OS

**Prerequisites:**
- Docker Desktop installed and running.

1. Clone this repository
2. Build the Docker image: `docker build -t profile_match_ai .`
3. Run the app using the batch script: `init.bat`

### Option 3: Pull pre-built Docker image from Docker Hub:
- `docker pull angelmacaroon4/profile_match_ai`
- `docker run -it -p 8501:8501 angelmacaroon4/profile_match_ai`
- Run the app using the batch script: `init.bat`

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
