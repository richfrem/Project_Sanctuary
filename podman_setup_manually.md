# Podman Setup Manually

```bash
#1. network already exists was created by mcp_gateway setup and verify it's there
podman compose -f docker-compose.yml up -d sanctuary_utils
podman images | grep sanctuary_utils
#2. add ollama
podman compose -f docker-compose.yml up -d ollama_model_mcp
podman images | grep ollama
#3. add vector db
podman compose -f docker-compose.yml up -d vector_db
podman images | grep chromadb
#4. add santuary_filesystem
# podman stop sanctuary_filesystem && podman rm sanctuary_filesystem
podman compose -f docker-compose.yml up -d sanctuary_filesystem
podman images | grep sanctuary_filesystem
#5. add sanctuary_network
podman compose -f docker-compose.yml up -d sanctuary_network
podman images | grep sanctuary_network
#6 add sanctuary_git
podman compose -f docker-compose.yml up -d sanctuary_git
podman images | grep sanctuary_git
#7 add sanctuary_domain
podman compose -f docker-compose.yml up -d sanctuary_domain
podman images | grep sanctuary_domain
#8 add sanctuary_cortex
podman compose -f docker-compose.yml up -d sanctuary_cortex
podman images | grep sanctuary_cortex

#check all images
podman containers
#check all containers
podman ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sanctuary

#test chroma script
python tests/mcp_servers/rag_cortex/inspect_chroma.py

#test ollama script
tests/mcp_servers/forge_llm/inspect_ollama.py

#register the fleet
python -m mcp_servers.gateway.fleet_setup
```
