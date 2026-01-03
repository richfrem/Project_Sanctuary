.PHONY: up down restart status verify build logs exec clean prune bootstrap install-env install-dev compile

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

# ----------------------------------------------------------------------------
# LOCAL ENVIRONMENT SETUP (ADR 073)
# ----------------------------------------------------------------------------

# Initial bootstrap for a fresh repository clone
# Usage: make bootstrap
bootstrap:
	@echo "🛡️  Bootstrapping Project Sanctuary environment..."
	@if [ ! -d ".venv" ]; then \
		echo "   Creating virtual environment..."; \
		python3 -m venv .venv; \
	fi
	@echo "   Installing core requirements..."
	@source .venv/bin/activate && pip install --upgrade pip pip-tools
	@$(MAKE) install-env
	@echo "✅ Bootstrap complete. Run 'source .venv/bin/activate' to begin."

# Install all runtime dependencies (Tier 1 & Tier 2)
# Usage: make install-env
install-env:
	@echo "📦 Installing shared core dependencies..."
	@if [ -f mcp_servers/requirements-core.txt ]; then \
		source .venv/bin/activate && pip install -r mcp_servers/requirements-core.txt; \
	else \
		echo "   ⚠️  mcp_servers/requirements-core.txt not found. Compiling..."; \
		$(MAKE) compile; \
		source .venv/bin/activate && pip install -r mcp_servers/requirements-core.txt; \
	fi
	@echo "📦 Installing service-specific requirements..."
	@for req in mcp_servers/gateway/clusters/*/requirements.txt; do \
		echo "   Installing $$req..."; \
		source .venv/bin/activate && pip install -r $$req; \
	done

# Install development & test dependencies (Tier 3)
# Usage: make install-dev
install-dev:
	@echo "🛠️  Installing development tools..."
	@source .venv/bin/activate && pip install -r requirements-dev.txt

# Re-compile all .in files to .txt lockfiles
# Usage: make compile
compile:
	@echo "🔐 Locking dependencies (pip-compile)..."
	@source .venv/bin/activate && pip install pip-tools
	@if [ -f mcp_servers/requirements-core.in ]; then \
		source .venv/bin/activate && pip-compile mcp_servers/requirements-core.in --output-file mcp_servers/requirements-core.txt; \
	fi
	@if [ -f requirements-dev.in ]; then \
		source .venv/bin/activate && pip-compile requirements-dev.in --output-file requirements-dev.txt; \
	fi
	@for req_in in mcp_servers/gateway/clusters/*/requirements.in; do \
		req_txt=$${req_in%.in}.txt; \
		echo "   Compiling $$req_in -> $$req_txt..."; \
		source .venv/bin/activate && pip-compile $$req_in --output-file $$req_txt; \
	done

# Upgrade all dependencies to latest versions
# Usage: make compile-upgrade
compile-upgrade:
	@echo "🔐 Upgrading dependency lockfiles (pip-compile --upgrade)..."
	@source .venv/bin/activate && pip install pip-tools
	@if [ -f mcp_servers/requirements-core.in ]; then \
		source .venv/bin/activate && pip-compile --upgrade mcp_servers/requirements-core.in --output-file mcp_servers/requirements-core.txt; \
	fi
	@if [ -f requirements-dev.in ]; then \
		source .venv/bin/activate && pip-compile --upgrade requirements-dev.in --output-file requirements-dev.txt; \
	fi
	@for req_in in mcp_servers/gateway/clusters/*/requirements.in; do \
		req_txt=$${req_in%.in}.txt; \
		echo "   Upgrading $$req_in -> $$req_txt..."; \
		source .venv/bin/activate && pip-compile --upgrade $$req_in --output-file $$req_txt; \
	done

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
