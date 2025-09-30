# Compliance Documentation

This directory contains compliance guidelines, regulatory requirements, and legal considerations for the Stoma Research Intelligence System.

## Contents

### Planned Documentation

- **Data Privacy Compliance** - GDPR, CCPA, and other privacy regulations
- **Terms of Service Compliance** - Respecting source platform ToS
- **Copyright & Fair Use** - Legal use of collected content
- **License Compliance** - Open source licenses and attribution
- **Data Retention Policies** - How long data is stored and why
- **User Data Protection** - If/when user accounts are added

## Compliance Framework

### Data Privacy

#### GDPR Compliance (European Union)
- **Right to Access**: Users can request their data
- **Right to Erasure**: Users can request data deletion
- **Data Minimization**: Only collect necessary data
- **Purpose Limitation**: Use data only for stated purposes
- **Storage Limitation**: Don't keep data longer than necessary

#### CCPA Compliance (California)
- **Right to Know**: Disclose what data is collected
- **Right to Delete**: Allow data deletion requests
- **Right to Opt-Out**: Opt-out of data selling (N/A - we don't sell data)

### Terms of Service Compliance

#### ArXiv
- ✅ Compliant: API usage within rate limits
- ✅ Compliant: Attribution provided
- ✅ Compliant: Non-commercial use (research purposes)

#### Reddit
- ⚠️ Review needed: API usage restrictions
- Required: Reddit API credentials and rate limiting
- Required: Respect subreddit rules

#### GitHub
- ✅ Compliant: API usage within rate limits
- ✅ Compliant: Public repository data only
- Required: GitHub API token for higher rate limits

#### HackerNews
- ✅ Compliant: Public API, no authentication required
- ✅ Compliant: Rate limiting implemented

#### SEC EDGAR
- ✅ Compliant: Required User-Agent header with contact info
- ✅ Compliant: 100ms rate limiting (SEC requirement)

### Copyright & Fair Use

#### Fair Use Considerations
1. **Purpose**: Research, analysis, and education
2. **Nature**: Factual/published content
3. **Amount**: Using portions for analysis, not full republication
4. **Effect**: No market harm to original sources

#### Attribution Requirements
- Credit original sources in reports
- Include source URLs and publication dates
- Maintain source metadata in database
- Display attribution in generated reports

### Web Scraping Ethics

#### Respectful Practices
- ✅ **robots.txt Compliance**: Always respect robots.txt directives
- ✅ **Rate Limiting**: 1-10 requests/second based on source
- ✅ **User-Agent**: Identify as Stoma with contact information
- ✅ **Caching**: Cache results to minimize repeat requests
- ✅ **Opt-Out**: Honor site requests to stop scraping

#### Configuration
```bash
# In .env
SCRAPER_RESPECT_ROBOTS_TXT=true
SCRAPER_RATE_LIMIT=1  # Requests per second
SCRAPER_USER_AGENT=Mozilla/5.0 (compatible; Stoma/0.1.0)
```

## Data Retention

### Current Policies

#### Raw Data
- **Retention**: 90 days for debugging and enrichment
- **Purpose**: Enable re-processing and error recovery
- **Cleanup**: Automated deletion after retention period

#### Processed Data
- **Retention**: Indefinite (until user deletion request)
- **Purpose**: Historical analysis and trend detection
- **Cleanup**: Available upon request

#### Logs
- **Retention**: 30 days
- **Purpose**: Debugging and monitoring
- **Cleanup**: Automated log rotation (5 files, 10MB each)

### Implementing Data Retention
```python
# Example cleanup script (to be implemented)
def cleanup_old_data():
    """Remove data older than retention period"""
    cutoff_date = datetime.now() - timedelta(days=90)
    # Delete raw data older than cutoff
    # Keep processed/analyzed data
```

## License Compliance

### Stoma License
- **Type**: MIT License (to be confirmed)
- **Permissions**: Commercial use, modification, distribution
- **Conditions**: Include copyright notice and license text
- **Limitations**: No warranty, no liability

### Third-Party Licenses

#### Python Dependencies
- Most dependencies use MIT, Apache 2.0, or BSD licenses
- Required: Include license texts in distributions
- Required: Maintain NOTICE files for Apache 2.0 dependencies

#### Data Sources
- ArXiv: Open access, non-exclusive license
- SEC EDGAR: Public domain (US government data)
- Academic papers: Varies by publisher (respect copyright)

## Risk Mitigation

### Legal Risks
- **Copyright Infringement**: Mitigated by fair use and attribution
- **ToS Violations**: Mitigated by respectful API usage and rate limiting
- **Privacy Violations**: Mitigated by data minimization and user controls

### Recommended Actions
- [ ] Consult legal counsel for commercial deployments
- [ ] Implement data deletion workflows
- [ ] Create privacy policy document
- [ ] Review and document all API usage compliance
- [ ] Implement audit logging for compliance tracking

## Audit & Reporting

### Compliance Audit Checklist
- [ ] Verify robots.txt compliance in web scraper
- [ ] Check rate limiting configuration for all sources
- [ ] Review data retention policies
- [ ] Audit third-party license compliance
- [ ] Document data processing activities
- [ ] Test data deletion workflows

### Reporting Requirements
- **Internal**: Quarterly compliance review
- **External**: Respond to data access/deletion requests within 30 days
- **Regulatory**: File required reports based on jurisdiction

---

**Status**: Documentation in progress
**Last Updated**: September 29, 2025

**⚠️ Disclaimer**: This documentation provides general guidelines. Consult legal counsel for specific compliance requirements in your jurisdiction.