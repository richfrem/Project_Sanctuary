#============================================
# Path: mcp_servers/gateway/clusters/sanctuary_network/models.py
# Purpose: Data definition layer for Network Cluster.
# Role: Data Layer
#============================================

from pydantic import BaseModel, Field

class FetchUrlRequest(BaseModel):
    url: str = Field(..., description="The URL to fetch content from via HTTP GET")

class SiteStatusRequest(BaseModel):
    url: str = Field(..., description="The URL to check status for via HTTP HEAD")
