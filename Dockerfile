FROM ollama/ollama:latest

# Expose Ollama API port
EXPOSE 11434

# Script to pull gemma3 model on first run
RUN echo '#!/bin/bash\n\
ollama serve &\n\
sleep 5\n\
ollama pull gemma3\n\
wait' > /start.sh && chmod +x /start.sh

ENTRYPOINT ["/bin/bash", "/start.sh"]
