import os 
import sys
from pathlib import Path
from rich import print
PROJECT_ROOT = Path(__file__).parent


DIRECTORIES = [
    # Core application layers (Separation of concerns)
    PROJECT_ROOT / "api",
    PROJECT_ROOT / "api" / "endpoints",
    PROJECT_ROOT / "api" / "models",
    PROJECT_ROOT / "api" / "controllers",
    
    # Domain logic (Business logic separated from infrastructure)
    PROJECT_ROOT / "core",
    PROJECT_ROOT / "core" / "agents",
    PROJECT_ROOT / "core" / "agents" / "transformer",
    PROJECT_ROOT / "core" / "agents" / "validator",
    PROJECT_ROOT / "core" / "orchestration",
    PROJECT_ROOT / "core" / "state_management",
    PROJECT_ROOT / "core" / "tools",
    PROJECT_ROOT / "core" / "tools" / "jolt",
    PROJECT_ROOT / "core" / "interfaces",  # For abstractions (Dependency Inversion)
    
    # Infrastructure layer
    PROJECT_ROOT / "infrastructure",
    PROJECT_ROOT / "infrastructure" / "repositories",
    PROJECT_ROOT / "infrastructure" / "services",
    
    # LLM Providers (Strategy pattern implementation)
    PROJECT_ROOT / "llm_providers",
    PROJECT_ROOT / "llm_providers" / "openai",
    PROJECT_ROOT / "llm_providers" / "anthropic",
    PROJECT_ROOT / "llm_providers" / "azure",
    PROJECT_ROOT / "llm_providers" / "local",
    PROJECT_ROOT / "llm_providers" / "interfaces",  # For provider interfaces
    
    # Supporting services
    PROJECT_ROOT / "services",
    PROJECT_ROOT / "services" / "logging",
    PROJECT_ROOT / "services" / "monitoring",
    PROJECT_ROOT / "services" / "authentication",
    
    # Configuration
    PROJECT_ROOT / "config",
    
    # Tests (Following the same structure as the main code)
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "tests" / "unit",
    PROJECT_ROOT / "tests" / "integration",
    PROJECT_ROOT / "tests" / "fixtures",
    
    # Deployment
    PROJECT_ROOT / "deployment",
    PROJECT_ROOT / "deployment" / "docker",
    
    # Data storage
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "data" / "states",
    PROJECT_ROOT / "data" / "transformations",
]

REQUIREMENTS = """

# Core dependencies - Python 3.11.0 compatible
fastapi==0.103.1
uvicorn[standard]==0.23.2
pydantic==2.7.1  
python-dotenv==1.0.0
starlette==0.27.0

# LangChain and related
langchain==0.0.285
langchain-experimental==0.0.9
langchainhub==0.1.13
langgraph==0.0.15

# LLM Providers
openai==1.3.0
anthropic==0.5.0
azure-identity==1.13.0
azure-mgmt-cognitiveservices==13.5.0

# JSON and JOLT support
jsonpath-ng==1.5.0
jmespath==1.0.1

# Logging and Elasticsearch
elasticsearch==8.9.0
elasticsearch-dsl==8.9.0
elasticsearch-logger==0.0.2
python-json-logger==2.0.7
loguru==0.7.0

# Monitoring
prometheus-client==0.17.1
prometheus-fastapi-instrumentator==6.1.0

# Testing
pytest==7.4.2
httpx==0.24.1
pytest-cov==4.1.0

# Deployment
docker==6.1.3
gunicorn==21.2.0

# Utils
PyYAML==6.0.1
typing-extensions==4.12.0  
requests==2.31.0
aiohttp==3.8.5
"""
ENV = """
    JOLT_API_HOST=sample-0.0.0.0
    JOLT_API_PORT=sample-8000
    JOLT_API_DEBUG=false

    JOLT_OPENAI_API_KEY=our-api-key
    JOLT_ANTHROPIC_API_KEY=our-api-key
    JOLT_AZURE_OPENAI_API_KEY=our-api-key
    JOLT_AZURE_OPENAI_ENDPOINT=our-endpoint

    JOLT_LOG_LEVEL=INFO
    JOLT_ELASTIC_ENABLED=false
    JOLT_ELASTIC_URL=our-elastic-url
    JOLT_ELASTIC_USERNAME=our-elastic-username
    JOLT_ELASTIC_PASSWORD=our-elastic-password

    JOLT_PROMETHEUS_PORT=sample-9090

"""

README = """
    An LLM-based agentic system that automatically generates JOLY transformation specifcations from input and output JSON examples.
    ##Overview

    The Jolt Transformation Engine uses a dual-agent architecture within the LangGraph framework:
    - The Transformer Agent generates JOLT specifications
    - The Validator Agent tests specifications with an embedded transformation tool

"""

def create_project_structure():
    """Create the project directory structure and files.
    """
    print("Creating JOLT Transformation Engine project structure...")
    for directory in DIRECTORIES:
        os.makedirs(directory,exist_ok=True)
        print(f"Created directory: {directory}")

    
    for directory in DIRECTORIES:

        if directory.parts[-1] not in ["data","deployment","docker","fixtures","states","transformations"]:
            init_file = directory/"__init__.py"
            print(init_file)
            with open(init_file,"w") as f:

                f.write("")
            print(f"Created file: {init_file}")
            f.close()
    
    with open(PROJECT_ROOT/"requirements.txt","w") as f:
        
        f.write(REQUIREMENTS)
    print(f"Created file: {PROJECT_ROOT}/requirements.txt")
    f.close()
    with open(PROJECT_ROOT/"README.md","w") as f:

        f.write(README)
    print(f"Created file: {PROJECT_ROOT}/README.md")
    f.close()


    with open(PROJECT_ROOT/".env","w") as f:

        f.write(ENV)
    print(f"Created file: {PROJECT_ROOT}/.env")
    f.close()

if __name__ == "__main__":

    create_project_structure()






