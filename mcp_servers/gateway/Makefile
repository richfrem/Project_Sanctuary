.PHONY: token db-migrate shell logs check-status

# Container name as per ADR 059
CONTAINER_NAME=gateway_gateway_1

# Default dev secret from podman-compose.yml
# In production, this would be passed via environment or vault
JWT_SECRET=sanctuary-mvp-secret

token:
	@echo "Generating Admin JWT Token..."
	@podman exec -it $(CONTAINER_NAME) python -m mcpgateway.utils.create_jwt_token --username admin@example.com --exp 10080 --secret $(JWT_SECRET) --algo HS256

db-migrate:
	@echo "Running Database Migrations..."
	@podman exec -it $(CONTAINER_NAME) alembic upgrade head

shell:
	@echo "Opening Shell in Gateway Container..."
	@podman exec -it $(CONTAINER_NAME) /bin/bash

logs:
	@podman logs $(CONTAINER_NAME)

check-status:
	@echo "Checking Container Status..."
	@podman ps -a --filter "name=$(CONTAINER_NAME)" --format "table {{.ID}} {{.Names}} {{.Status}}"
