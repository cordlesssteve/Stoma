# Stoma Repository Structure Migration Plan

**Status**: ACTIVE
**Created**: 2025-09-29
**Last Updated**: 2025-09-29
**Purpose**: Comprehensive migration plan to achieve REPOSITORY_STRUCTURE_STANDARD compliance
**Priority**: CRITICAL

---

## 🚨 Executive Summary

Stoma is currently **non-compliant** with the REPOSITORY_STRUCTURE_STANDARD and requires comprehensive restructuring to achieve standardization across the development ecosystem.

### **Critical Issues Identified**
- ❌ Source code in `/stoma/` instead of standard `/src/`
- ❌ Configuration files scattered in root instead of `/config/`
- ❌ Scripts mixed with source instead of organized `/scripts/`
- ❌ Missing deployment infrastructure (`/deployment/`)
- ❌ No CI/CD automation (`.github/`)

### **Business Impact**
- **Development Efficiency**: Non-standard structure increases onboarding time
- **Tooling Compatibility**: Cannot use universal CI/CD pipeline templates
- **Maintenance Overhead**: Inconsistent structure complicates automation
- **Claude Code Integration**: Sub-optimal integration with standard tooling

---

## 📊 Current vs Target Structure Analysis

### **Current Structure (Non-Compliant)**
```
Stoma/
├── stoma/                  ❌ Should be /src/
├── config.yaml               ❌ Should be /config/environments/
├── config_local.yaml         ❌ Should be /config/environments/
├── dev-scripts/              ❌ Should be /scripts/utilities/
├── tests/                    ✅ Correct location
├── docs/                     ✅ Correct location
├── external/                 ❓ Needs evaluation
├── reports/                  ❓ Should be in /src/ or data/
├── production_pipeline_data/ ❓ Should be in data/ or temp/
└── [Missing directories]     ❌ deployment/, .github/
```

### **Target Structure (Compliant)**
```
Stoma/
├── src/                      ← Main source code
│   └── stoma/             ← Current /stoma/ content
├── tests/                    ← Already correct
├── docs/                     ← Already correct
├── scripts/                  ← Automation & utilities
├── config/                   ← Configuration management
├── deployment/               ← Docker & infrastructure
├── .github/                  ← CI/CD workflows
├── data/                     ← Data storage & pipelines
└── temp/                     ← Temporary files
```

---

## 🗓️ Migration Phases

### **Phase 1: Source Code Organization**
**Estimated Time**: 2-3 hours
**Risk Level**: HIGH (affects all imports)

#### **1.1 Create Standard Directory Structure**
```bash
mkdir -p src/
mkdir -p config/environments/
mkdir -p scripts/{data,monitoring,security,utilities}
mkdir -p deployment/{docker,kubernetes,scripts,environments}
mkdir -p .github/{workflows,ISSUE_TEMPLATE}
```

#### **1.2 Move Source Code**
```bash
# Move main source code
mv stoma/ src/stoma/

# Update package structure
# This requires updating all import statements
```

#### **1.3 Update Import Statements**
**Critical**: All Python imports need updating
```python
# OLD: from stoma.pipeline import DataPipeline
# NEW: from src.stoma.pipeline import DataPipeline
```

**Files Requiring Updates**:
- CLI entry points
- Test files
- Setup.py/pyproject.toml
- All internal imports

#### **1.4 Update Configuration References**
- CLI command paths
- Test discovery paths
- Package installation paths
- Documentation references

### **Phase 2: Configuration Management**
**Estimated Time**: 1-2 hours
**Risk Level**: MEDIUM

#### **2.1 Reorganize Configuration Files**
```bash
# Move configurations to standard structure
mv config.yaml config/environments/production.yaml
mv config_local.yaml config/environments/development.yaml
mv example_config.yaml config/defaults/system.yaml

# Create environment template
cp config/environments/development.yaml .env.template
```

#### **2.2 Create Environment-Specific Configs**
```yaml
# config/environments/development.yaml
database:
  url: "postgresql://localhost:5433/stoma_dev"

# config/environments/production.yaml
database:
  url: "${DATABASE_URL}"

# config/environments/testing.yaml
database:
  url: "sqlite:///test.db"
```

#### **2.3 Update Configuration Loading**
Update code to load from new config structure:
```python
# OLD: config = load_config("config.yaml")
# NEW: config = load_config(f"config/environments/{env}.yaml")
```

### **Phase 3: Scripts & Automation**
**Estimated Time**: 2-3 hours
**Risk Level**: LOW

#### **3.1 Reorganize Existing Scripts**
```bash
# Move development scripts to utilities
mv dev-scripts/* scripts/utilities/

# Create required automation scripts
```

#### **3.2 Create Required Automation Scripts**
**scripts/setup.sh**:
```bash
#!/bin/bash
# Environment setup automation
pip install -r requirements.txt
ollama pull llama3.1:latest
python3 -m pytest tests/ --setup-only
```

**scripts/dev.sh**:
```bash
#!/bin/bash
# Development server
export ENVIRONMENT=development
python3 -m src.stoma.cli.main "$@"
```

**scripts/test.sh**:
```bash
#!/bin/bash
# Run all tests
python3 -m pytest tests/ -v
```

**scripts/lint.sh**:
```bash
#!/bin/bash
# Code quality checks
ruff check src/
mypy src/stoma/
```

#### **3.3 Create Build & Deploy Scripts**
**scripts/build.sh**:
```bash
#!/bin/bash
# Build/package process
python3 -m build
docker build -t stoma:latest .
```

**scripts/deploy.sh**:
```bash
#!/bin/bash
# Deployment automation
docker-compose -f deployment/docker/docker-compose.yml up -d
```

### **Phase 4: Deployment Infrastructure**
**Estimated Time**: 1-2 hours
**Risk Level**: LOW

#### **4.1 Create Docker Configuration**
**deployment/docker/Dockerfile**:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY config/ ./config/
CMD ["python3", "-m", "src.stoma.cli.main"]
```

**deployment/docker/docker-compose.yml**:
```yaml
version: '3.8'
services:
  stoma:
    build:
      context: ../..
      dockerfile: deployment/docker/Dockerfile
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./data:/app/data
```

#### **4.2 Create Environment-Specific Deployments**
```bash
deployment/
├── environments/
│   ├── staging/
│   │   └── docker-compose.staging.yml
│   └── production/
│       └── docker-compose.prod.yml
└── scripts/
    ├── deploy-staging.sh
    └── deploy-production.sh
```

### **Phase 5: CI/CD Automation**
**Estimated Time**: 1-2 hours
**Risk Level**: LOW

#### **5.1 Create GitHub Workflows**
**.github/workflows/ci.yml**:
```yaml
name: CI Pipeline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python3 -m pytest tests/
      - name: Run linting
        run: |
          ruff check src/
```

#### **5.2 Create Issue Templates**
**.github/ISSUE_TEMPLATE/bug_report.md**
**.github/ISSUE_TEMPLATE/feature_request.md**
**.github/PULL_REQUEST_TEMPLATE.md**

### **Phase 6: Data & Reports Organization**
**Estimated Time**: 1 hour
**Risk Level**: LOW

#### **6.1 Organize Data Directories**
```bash
# Create data structure
mkdir -p data/{raw,processed,reports,exports}

# Move existing data
mv reports/ data/reports/
mv production_pipeline_data/ data/pipeline/
mv formatted_reports/ data/exports/
```

#### **6.2 Update Data Access Patterns**
Update code to use new data paths:
```python
# OLD: reports_dir = "reports/"
# NEW: reports_dir = "data/reports/"
```

---

## ⚠️ Migration Risks & Mitigation

### **High Risk Items**

#### **Import Statement Updates**
**Risk**: Broken imports causing application failure
**Mitigation**:
- Create comprehensive test suite for import validation
- Use automated refactoring tools where possible
- Test in isolated environment first

#### **Configuration Loading**
**Risk**: Application unable to load configuration
**Mitigation**:
- Maintain backward compatibility during transition
- Implement graceful fallback to old config locations
- Test all environment configurations

#### **CLI Command Paths**
**Risk**: CLI commands fail due to path changes
**Mitigation**:
- Update setup.py/pyproject.toml entry points
- Test all CLI commands after migration
- Update documentation with new command structure

### **Medium Risk Items**

#### **External Dependencies**
**Risk**: External tools expecting old structure
**Mitigation**:
- Audit external integrations
- Update any hardcoded paths
- Provide migration notices

#### **Development Environment**
**Risk**: Broken development workflows
**Mitigation**:
- Update IDE configurations
- Test development scripts
- Update team documentation

---

## 🧪 Testing Strategy

### **Pre-Migration Testing**
1. **Backup Current State**: Create complete repository backup
2. **Import Mapping**: Document all current import statements
3. **Configuration Audit**: Document all configuration dependencies
4. **Integration Testing**: Verify all current functionality works

### **Migration Testing**
1. **Phase-by-Phase Validation**: Test each phase independently
2. **Integration Testing**: Verify cross-component functionality
3. **CLI Testing**: Test all CLI commands and workflows
4. **Configuration Testing**: Verify all environments load correctly

### **Post-Migration Validation**
1. **Full Functionality Test**: Complete end-to-end workflow testing
2. **Performance Baseline**: Ensure no performance degradation
3. **Documentation Validation**: Verify all documentation is updated
4. **Compliance Check**: Run REPOSITORY_STRUCTURE_STANDARD validation

---

## 📋 Migration Checklist

### **Phase 1: Source Code** ⏳
- [ ] Create `/src/` directory structure
- [ ] Move `/stoma/` to `/src/stoma/`
- [ ] Update all import statements in source code
- [ ] Update all import statements in tests
- [ ] Update setup.py/pyproject.toml entry points
- [ ] Update CLI command references
- [ ] Test source code imports
- [ ] Test CLI functionality

### **Phase 2: Configuration** ⏳
- [ ] Create `/config/` directory structure
- [ ] Move configuration files to environments
- [ ] Create environment-specific configs
- [ ] Update configuration loading code
- [ ] Create `.env.template`
- [ ] Test configuration loading
- [ ] Test all environments

### **Phase 3: Scripts** ⏳
- [ ] Create `/scripts/` directory structure
- [ ] Move dev-scripts to utilities
- [ ] Create setup.sh automation
- [ ] Create dev.sh development script
- [ ] Create test.sh testing script
- [ ] Create lint.sh quality checks
- [ ] Create build.sh packaging
- [ ] Create deploy.sh deployment
- [ ] Test all automation scripts

### **Phase 4: Deployment** ⏳
- [ ] Create `/deployment/` directory structure
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Create environment-specific deployments
- [ ] Create deployment scripts
- [ ] Test Docker build
- [ ] Test container deployment

### **Phase 5: CI/CD** ⏳
- [ ] Create `.github/` directory structure
- [ ] Create CI workflow
- [ ] Create issue templates
- [ ] Create pull request template
- [ ] Test GitHub Actions
- [ ] Verify workflow execution

### **Phase 6: Data** ⏳
- [ ] Create `/data/` directory structure
- [ ] Move reports to data/reports/
- [ ] Move pipeline data to data/pipeline/
- [ ] Update data access patterns
- [ ] Test data access functionality

### **Final Validation** ⏳
- [ ] Run REPOSITORY_STRUCTURE_STANDARD compliance check
- [ ] Complete end-to-end functionality test
- [ ] Update all documentation
- [ ] Verify performance baseline
- [ ] Complete migration sign-off

---

## 🎯 Success Criteria

### **Compliance Achievement**
- ✅ All required directories present per REPOSITORY_STRUCTURE_STANDARD
- ✅ All mandatory root files in correct locations
- ✅ Consistent naming conventions throughout project
- ✅ Proper separation of concerns (source, tests, config, deployment)

### **Functionality Preservation**
- ✅ All existing CLI commands work correctly
- ✅ All tests pass without modification
- ✅ All integrations (OpenDeepResearch, Ollama) function correctly
- ✅ All analysis workflows complete successfully

### **Automation Enhancement**
- ✅ Standard automation scripts functional
- ✅ CI/CD pipeline operational
- ✅ Docker deployment working
- ✅ Development workflow streamlined

---

## 📞 Post-Migration Actions

### **Documentation Updates**
1. Update CURRENT_STATUS.md with compliance achievement
2. Update ACTIVE_PLAN.md with new development priorities
3. Update README.md with new structure overview
4. Update CLAUDE.md with new development commands

### **Team Communication**
1. Notify team of structure changes
2. Update development environment setup guides
3. Provide migration summary and new workflows
4. Schedule structure compliance review

### **Continuous Improvement**
1. Monitor for any missed migration items
2. Collect feedback on new structure usability
3. Iterate on automation scripts based on usage
4. Plan for future compliance maintenance

---

**Migration Lead**: Claude Code
**Review Required**: Development Team Lead
**Estimated Total Time**: 8-12 hours across 6 phases
**Success Measurement**: Full REPOSITORY_STRUCTURE_STANDARD compliance + preserved functionality