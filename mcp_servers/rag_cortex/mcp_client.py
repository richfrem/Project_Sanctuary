#!/usr/bin/env python3
"""
RAG Cortex MCP Client (Protocol 87 Router)
=====================================

Purpose:
    Client for routing Protocol 87 queries to specialized MCPs.
    Implements the MCP composition pattern from ADR 039.
    Routes queries based on intent (e.g. Protocol, Chronicle, Task).

Layer: Client / Router

Key Classes:
    - MCPClient: Main routing logic
        - __init__(project_root)
        - route_query(scope, intent, constraints, query_data)
        - _query_protocols(intent, constraints, query_data)
        - _query_chronicles(intent, constraints, query_data)
        - _query_tasks(intent, constraints, query_data)
        - _query_code(intent, constraints, query_data)
        - _query_adrs(intent, constraints, query_data)
"""

from typing import Dict, Any, List, Optional
import json


class MCPClient:
    #============================================
    # Class: MCPClient
    # Purpose: Client for routing queries to specialized MCP servers.
    # Patterns: Router / Composition
    #============================================

    def __init__(self, project_root: str):
        #============================================
        # Method: __init__
        # Purpose: Initialize MCP client.
        # Args:
        #   project_root: Path to project root directory
        #============================================
        self.project_root = project_root
    
    def route_query(
        self,
        scope: str,
        intent: str,
        constraints: str,
        query_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        #============================================
        # Method: route_query
        # Purpose: Route query to appropriate MCP based on scope.
        # Args:
        #   scope: Query scope (Protocols, Living_Chronicle, tasks, Code, ADRs)
        #   intent: Query intent (RETRIEVE, SUMMARIZE, CROSS_COMPARE, VERIFY)
        #   constraints: Query constraints string
        #   query_data: Full parsed query data
        # Returns: List of results from target MCP
        #============================================
        # Route based on scope
        if scope == "Protocols":
            return self._query_protocols(intent, constraints, query_data)
        elif scope == "Living_Chronicle":
            return self._query_chronicles(intent, constraints, query_data)
        elif scope == "tasks":
            return self._query_tasks(intent, constraints, query_data)
        elif scope == "Code":
            return self._query_code(intent, constraints, query_data)
        elif scope == "ADRs":
            return self._query_adrs(intent, constraints, query_data)
        else:
            # Unknown scope - return empty for now
            # In production, this would fallback to vector DB
            return []
    
    def _query_protocols(
        self,
        intent: str,
        constraints: str,
        query_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        #============================================
        # Method: _query_protocols
        # Purpose: Query Protocol MCP.
        # Args:
        #   intent: Query intent
        #   constraints: Query constraints
        #   query_data: Full query data
        # Returns: List of tool results
        #============================================
        try:
            from mcp_servers.protocol.operations import ProtocolOperations
            
            ops = ProtocolOperations(self.project_root)
            
            if intent == "RETRIEVE":
                # Extract protocol number from constraints
                # Format: Name="Protocol 101" → 101
                if "Name=" in constraints or "name=" in constraints.lower():
                    # Parse protocol number
                    name_part = constraints.lower().split("name=")[1]
                    # Extract number
                    import re
                    numbers = re.findall(r'\d+', name_part)
                    if numbers:
                        number = int(numbers[0])
                        result = ops.get_protocol(number)
                        return [{
                            "source": "Protocol MCP",
                            "source_path": f"01_PROTOCOLS/{number:03d}_*.md",
                            "content": result,
                            "mcp_tool": "protocol_get"
                        }]
            
            elif intent == "SUMMARIZE":
                # List all protocols
                results = ops.list_protocols()
                return [{
                    "source": "Protocol MCP",
                    "content": results,
                    "mcp_tool": "protocol_list"
                }]
            
        except Exception as e:
            return [{
                "source": "Protocol MCP",
                "error": str(e),
                "status": "error"
            }]
        
        return []
    
    def _query_chronicles(
        self,
        intent: str,
        constraints: str,
        query_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        #============================================
        # Method: _query_chronicles
        # Purpose: Query Chronicle MCP.
        # Args:
        #   intent: Query intent
        #   constraints: Query constraints
        #   query_data: Full query data
        # Returns: List of tool results
        #============================================
        try:
            from mcp_servers.chronicle.operations import ChronicleOperations
            
            ops = ChronicleOperations(self.project_root)
            
            if intent == "RETRIEVE":
                # Extract entry number
                # Format: Anchor=245 or Entry=245
                if "Anchor=" in constraints or "Entry=" in constraints:
                    import re
                    numbers = re.findall(r'\d+', constraints)
                    if numbers:
                        entry_num = int(numbers[0])
                        result = ops.get_entry(entry_num)
                        return [{
                            "source": "Chronicle MCP",
                            "source_path": f"00_CHRONICLE/ENTRIES/{entry_num:03d}_*.md",
                            "content": result,
                            "mcp_tool": "chronicle_get_entry"
                        }]
            
            elif intent == "SUMMARIZE":
                # List recent entries
                # Extract limit from constraints if present
                limit = 10
                if "Timeframe=" in constraints:
                    # Parse range like "Entries(240-245)"
                    import re
                    numbers = re.findall(r'\d+', constraints)
                    if len(numbers) >= 2:
                        limit = int(numbers[1]) - int(numbers[0]) + 1
                
                results = ops.list_entries(limit=limit)
                return [{
                    "source": "Chronicle MCP",
                    "content": results,
                    "mcp_tool": "chronicle_list_entries"
                }]
            
        except Exception as e:
            return [{
                "source": "Chronicle MCP",
                "error": str(e),
                "status": "error"
            }]
        
        return []
    
    def _query_tasks(
        self,
        intent: str,
        constraints: str,
        query_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        #============================================
        # Method: _query_tasks
        # Purpose: Query Task MCP.
        # Args:
        #   intent: Query intent
        #   constraints: Query constraints
        #   query_data: Full query data
        # Returns: List of tool results
        #============================================
        try:
            from mcp_servers.task.operations import TaskOperations
            
            ops = TaskOperations(self.project_root)
            
            if intent == "RETRIEVE":
                # Extract task number
                if "Number=" in constraints or "Task=" in constraints:
                    import re
                    numbers = re.findall(r'\d+', constraints)
                    if numbers:
                        task_num = int(numbers[0])
                        result = ops.get_task(task_num)
                        return [{
                            "source": "Task MCP",
                            "source_path": f"tasks/*/{task_num:03d}_*.md",
                            "content": result,
                            "mcp_tool": "get_task"
                        }]
            
            elif intent == "SUMMARIZE":
                # List tasks by status
                status = None
                if "Status=" in constraints:
                    # Extract status value
                    status_part = constraints.split("Status=")[1]
                    status = status_part.strip('"').strip("'").split()[0]
                
                results = ops.list_tasks(status=status)
                return [{
                    "source": "Task MCP",
                    "content": results,
                    "mcp_tool": "list_tasks"
                }]
            
        except Exception as e:
            return [{
                "source": "Task MCP",
                "error": str(e),
                "status": "error"
            }]
        
        return []
    
    def _query_code(
        self,
        intent: str,
        constraints: str,
        query_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        #============================================
        # Method: _query_code
        # Purpose: Query Code MCP.
        # Args:
        #   intent: Query intent
        #   constraints: Query constraints
        #   query_data: Full query data
        # Returns: List of tool results
        #============================================
        try:
            from mcp_servers.code.operations import CodeOperations
            
            ops = CodeOperations(self.project_root)
            
            if intent == "RETRIEVE":
                # Search code content
                # Extract search query from constraints
                query = constraints.strip('"').strip("'")
                results = ops.search_content(query)
                return [{
                    "source": "Code MCP",
                    "content": results,
                    "mcp_tool": "code_search_content"
                }]
            
        except Exception as e:
            return [{
                "source": "Code MCP",
                "error": str(e),
                "status": "error"
            }]
        
        return []
    
    def _query_adrs(
        self,
        intent: str,
        constraints: str,
        query_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        #============================================
        # Method: _query_adrs
        # Purpose: Query ADR MCP.
        # Args:
        #   intent: Query intent
        #   constraints: Query constraints
        #   query_data: Full query data
        # Returns: List of tool results
        #============================================
        try:
            from mcp_servers.adr.operations import ADROperations
            
            ops = ADROperations(self.project_root)
            
            if intent == "RETRIEVE":
                # Extract ADR number
                if "Number=" in constraints or "ADR=" in constraints:
                    import re
                    numbers = re.findall(r'\d+', constraints)
                    if numbers:
                        adr_num = int(numbers[0])
                        result = ops.get_adr(adr_num)
                        return [{
                            "source": "ADR MCP",
                            "source_path": f".agent/adr/{adr_num:03d}_*.md",
                            "content": result,
                            "mcp_tool": "adr_get"
                        }]
            
            elif intent == "SUMMARIZE":
                # List all ADRs
                results = ops.list_adrs()
                return [{
                    "source": "ADR MCP",
                    "content": results,
                    "mcp_tool": "adr_list"
                }]
            
        except Exception as e:
            return [{
                "source": "ADR MCP",
                "error": str(e),
                "status": "error"
            }]
        
        return []
