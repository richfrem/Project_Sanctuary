.PHONY: up down restart status verify build logs exec clean prune

# Unified Fleet Operations Makefile (ADR 065 v1.3)
# "The Iron Root" - Single Source of Truth for Fleet Management
#
# LOCATION: Project Root
#
# PRECONDITIONS:
#   1. Podman (4.x+) installed and running (`podman system service` active).
#   2. 'docker-compose.yml' present in PWD defining the 8-container fleet.
#   3. '.env' file populated with MCPGATEWAY_BEARER_TOKEN and gateway URLs.
#   4. 'sanctuary_gateway' running externally (Port 4444).
#
# OUTPUTS:
#   - Physical: 8 Podman containers running/restarted.
#   - Logical: 'mcp_servers/gateway/fleet_registry.json' updated with current discovery data.
#   - Stdout: Logs for build, deployment, pulse checks, and orchestration handshake.

# 1. Environment Handling
# Incorporate .env vars for usage in targets
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Default shell
SHELL := /bin/bash

# Configuration
COMPOSE_FILE := docker-compose.yml
GATEWAY_URL ?= https://localhost:4444

# ----------------------------------------------------------------------------
# CORE LIFECYCLE TARGETS
# ----------------------------------------------------------------------------

# Deploy the entire fleet
# Usage: make up [force=true]
up:
	@echo "🚀 [1/4] Checking Pre-requisites..."
	@# Check Gateway Health (warn only, as it might be starting up in a separate stack)
	@if curl -k -s -f -o /dev/null "$(GATEWAY_URL)/health"; then \
		echo "   ✅ Gateway is reachable."; \
	else \
		echo "   ⚠️  Gateway unreachable at $(GATEWAY_URL). Orchestration may fail."; \
	fi

	@echo "📦 [2/4] Deploying Physical Containers..."
	podman compose -f $(COMPOSE_FILE) up -d $(if $(force),--build,)

	@echo "💓 [3/4] Waiting for Fleet Pulse (Health check)..."
	@./scripts/wait_for_pulse.sh

	@echo "🎼 [4/4] Fleet Registration & Discovery (Clean + Register)..."
	@if [ -f .env ]; then \
		set -a && source .env && set +a && python3 -m mcp_servers.gateway.fleet_setup; \
	else \
		python3 -m mcp_servers.gateway.fleet_setup; \
	fi
	@echo "✅ Fleet Deployed & Registered."

# Stop the fleet
down:
	@echo "🛑 Stopping Fleet..."
	podman compose -f $(COMPOSE_FILE) down

# Restart specific service or all
# Usage: make restart [TARGET=sanctuary_cortex]
restart:
	@echo "🔄 Restarting $(if $(TARGET),$(TARGET),all services)..."
	@if [ -n "$(TARGET)" ]; then \
		podman compose -f $(COMPOSE_FILE) stop $(TARGET); \
		podman compose -f $(COMPOSE_FILE) up -d $(TARGET); \
	else \
		make down; \
		make up; \
	fi
	@echo "🎼 Re-triggering Orchestration..."
	@sleep 2
	@if [ -f .env ]; then \
		set -a && source .env && set +a && python3 -m mcp_servers.gateway.fleet_orchestrator; \
	else \
		python3 -m mcp_servers.gateway.fleet_orchestrator; \
	fi

# ----------------------------------------------------------------------------
# OBSERVABILITY & MAINTENANCE
# ----------------------------------------------------------------------------

# Show status of infrastructure and registration
status:
	@echo "\n📊 Physical Fleet Status (Podman):"
	@podman ps --filter "name=sanctuary" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	
	@echo "\n📜 Logical Fleet Status (Registry):"
	@if [ -f mcp_servers/gateway/fleet_registry.json ]; then \
		echo "   Found registry file."; \
		grep -E "status|tool_count" mcp_servers/gateway/fleet_registry.json | head -n 10 || echo "   (Empty or invalid JSON)"; \
	else \
		echo "   ⚠️  Registry not found (Run 'make up')."; \
	fi

# View logs
# Usage: make logs [TARGET=sanctuary_vector_db]
logs:
	podman compose -f $(COMPOSE_FILE) logs -f $(TARGET)

# Interactive shell
# Usage: make exec TARGET=sanctuary_git
exec:
	@if [ -z "$(TARGET)" ]; then echo "❌ Error: Must specify TARGET (e.g., make exec TARGET=sanctuary_git)"; exit 1; fi
	podman compose -f $(COMPOSE_FILE) exec $(TARGET) /bin/sh

# Build images without starting
build:
	podman compose -f $(COMPOSE_FILE) build

# Clean up volumes and images
clean:
	@echo "⚠️  WARNING: This will delete all fleet data (ChromeDB, etc)."
	@read -p "Are you sure? [y/N] " ans && [ $${ans:-N} = y ]
	podman compose -f $(COMPOSE_FILE) down -v --rmi all

# Safe prune (removes stopped containers, build cache, dangling images - NOT volumes)
prune:
	@echo "🧹 Pruning build cache and stopped containers..."
	podman container prune -f
	podman image prune -f
	podman builder prune -f
	@echo "✅ Prune complete. Data volumes preserved."

# ----------------------------------------------------------------------------
# VERIFICATION
# ----------------------------------------------------------------------------

verify:
	@echo "🧪 Running Connectivity Tests..."
	pytest mcp_servers/gateway/test_gateway_blackbox.py -v
