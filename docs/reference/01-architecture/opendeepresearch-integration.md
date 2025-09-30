# OpenDeepResearch Integration Architecture

**Status**: ACTIVE
**Last Updated**: 2025-09-29
**Version**: 1.0

## Overview

This document describes the architecture and implementation of the OpenDeepResearch integration with Stoma, providing multi-agent research capabilities through a git submodule architecture.

## Architecture Components

### Git Submodule Structure
```
/external/open_deep_research/  # Git submodule
├── src/legacy/                # Legacy multi-agent implementation (WORKING)
├── src/main/                  # Main LangGraph workflows (TIMEOUT ISSUES)
└── requirements.txt           # Dependencies
```

### Bridge Pattern Implementation

**Location**: `stoma/integrations/ollama_deep_research_bridge.py`

```python
class OllamaDeepResearchBridge:
    """Bridge between Stoma and OpenDeepResearch using legacy multi-agent architecture."""
    
    def __init__(self, model_name: str = "llama3.1:latest", storage_manager: Optional[ReportStorageManager] = None):
        self.model_name = model_name
        self.storage_manager = storage_manager or ReportStorageManager()
```

### Multi-Agent Workflow

1. **Supervisor Agent**: Coordinates research tasks and manages workflow
2. **Researcher Agents**: Perform parallel research using tools
3. **Report Assembly**: Combines agent outputs into structured reports

## Technical Implementation

### Tool Calling Configuration

**Working Solution**: ChatOllama from `langchain-ollama`
```python
from langchain_ollama import ChatOllama

# WORKS with tool calling
chat_model = ChatOllama(model="llama3.1:latest", temperature=0)

# FAILS with tool calling
chat_model = init_chat_model("ollama:llama3.1:latest")
```

### Key Fixes Applied

1. **Removed parallel_tool_calls parameter**: Caused "unexpected keyword argument" error
2. **Fixed variable naming conflicts**: Renamed 'tool' to 'tool_instance'
3. **Added proper error handling**: Graceful degradation when tools fail
4. **Implemented recursion limits**: Prevent infinite agent loops

## Integration Points

### CLI Commands

1. `stoma deep-research ollama-research` - Direct topic research
2. `stoma deep-research analyze-papers` - Multi-paper analysis  
3. `stoma deep-research citation-research` - Research with citations

### Report Storage

Automatic integration with Stoma's `ReportStorageManager`:
- JSON reports saved to structured directories
- Metadata tracking and indexing
- Search and retrieval capabilities

## Known Issues & Solutions

### Main LangGraph Workflow Timeout

**Issue**: OpenDeepResearch's main implementation times out indefinitely with Ollama
**Solution**: Use legacy multi-agent implementation which works reliably

### Tool Binding Errors

**Issue**: `init_chat_model("ollama:...")` doesn't support tool binding
**Solution**: Use `ChatOllama` from `langchain-ollama` package

### Recursion Limits

**Issue**: Graph recursion limit of 25 reached
**Solution**: Modified `research_agent_should_continue` to properly end workflows

## Performance Characteristics

- **Model**: llama3.1:latest (8GB)
- **Response Time**: ~30-60 seconds for comprehensive research
- **Tool Calls**: ArXiv search, web search, document analysis
- **Output Quality**: Structured reports with introduction, sections, conclusions

## Future Enhancements

1. **Main Workflow Fix**: Investigate timeout issues in primary LangGraph implementation
2. **Tool Expansion**: Add more research tools (SEC filings, patent search)
3. **Model Optimization**: Support for larger models (70B+)
4. **Parallel Processing**: Multiple research topics simultaneously