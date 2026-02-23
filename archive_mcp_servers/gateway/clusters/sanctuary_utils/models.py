#============================================
# Path: mcp_servers/gateway/clusters/sanctuary_utils/models.py
# Purpose: Data definition layer for Utility Cluster.
# Role: Data Layer
#============================================

from typing import Optional, List, Union
from pydantic import BaseModel, Field

# Time Models
class TimeCurrentRequest(BaseModel):
    timezone_name: str = Field("UTC", description="Timezone name (default: UTC)")

# Calculator Models
class CalcExpressionRequest(BaseModel):
    expression: str = Field(..., description="Math expression to evaluate (e.g., '2 + 2', 'sqrt(16)')")

class CalcBinaryRequest(BaseModel):
    a: Union[int, float] = Field(..., description="First number")
    b: Union[int, float] = Field(..., description="Second number")

# UUID Models
class UUIDValidateRequest(BaseModel):
    uuid_string: str = Field(..., description="UUID string to validate")

# String Models
class StringSingleRequest(BaseModel):
    text: str = Field(..., description="Text to process")

class StringReplaceRequest(BaseModel):
    text: str = Field(..., description="Original text")
    old: str = Field(..., description="Substring to replace")
    new: str = Field(..., description="Replacement substring")
