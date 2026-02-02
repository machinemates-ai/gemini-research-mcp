#!/usr/bin/env python3
"""
Query Gemini API usage directly from Google Cloud.

This script fetches actual usage data from Google Cloud APIs,
not relying on locally stored sessions.

Usage:
    python gemini_usage.py                    # Last 30 days
    python gemini_usage.py --days 7           # Last 7 days
    python gemini_usage.py --start 2026-01-01 # Since specific date
    
Requirements:
    pip install google-cloud-billing google-cloud-monitoring google-auth
    
    Or with uv:
    uv run --with google-cloud-billing --with google-cloud-monitoring python gemini_usage.py

Authentication:
    gcloud auth application-default login
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any

# Google Cloud clients
try:
    from google.cloud import monitoring_v3
    from google.cloud.billing import budgets_v1
    from google.cloud import billing_v1
    HAS_CLOUD_LIBS = True
except ImportError:
    HAS_CLOUD_LIBS = False

try:
    import google.auth
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    import requests
    HAS_AUTH = True
except ImportError:
    HAS_AUTH = False


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ID = "gen-lang-client-0202182328"
BILLING_ACCOUNT = "01D2F4-3F2B44-2B0851"

# Gemini API pricing (Feb 2026)
PRICING = {
    "gemini-2.0-flash": {
        "input_per_1m": 0.10,
        "output_per_1m": 0.40,
    },
    "gemini-2.0-flash-thinking": {
        "input_per_1m": 0.10,
        "output_per_1m": 0.40,
    },
    "gemini-3-pro": {  # Deep Research
        "input_per_1m": 2.00,
        "output_per_1m": 12.00,
    },
}


# =============================================================================
# Method 1: BigQuery Export (Most Accurate)
# =============================================================================

def get_usage_from_bigquery(
    project_id: str = PROJECT_ID,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """
    Query Gemini API usage from BigQuery billing export.
    
    Requires billing export to be enabled:
    https://cloud.google.com/billing/docs/how-to/export-data-bigquery
    
    This is the MOST ACCURATE method as it uses actual billing data.
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        print("❌ google-cloud-bigquery not installed")
        print("   pip install google-cloud-bigquery")
        return []
    
    client = bigquery.Client(project=project_id)
    
    # Adjust dates
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Query the billing export table
    # Note: You need to set up billing export first
    # The dataset/table name depends on your setup
    query = f"""
    SELECT
        service.description as service,
        sku.description as sku,
        usage_start_time,
        usage_end_time,
        usage.amount as usage_amount,
        usage.unit as usage_unit,
        cost,
        currency,
        project.id as project_id
    FROM `{project_id}.billing_export.gcp_billing_export_v1_{BILLING_ACCOUNT.replace('-', '_')}`
    WHERE 
        service.description LIKE '%Generative%'
        OR service.description LIKE '%Vertex AI%'
        OR service.description LIKE '%Gemini%'
    AND DATE(usage_start_time) >= '{start_date}'
    AND DATE(usage_end_time) <= '{end_date}'
    ORDER BY usage_start_time DESC
    """
    
    try:
        results = client.query(query).result()
        return [dict(row) for row in results]
    except Exception as e:
        print(f"⚠️ BigQuery query failed: {e}")
        print("   Make sure billing export is enabled")
        return []


# =============================================================================
# Method 2: Cloud Monitoring API
# =============================================================================

def get_usage_from_monitoring(
    project_id: str = PROJECT_ID,
    days: int = 30,
) -> dict[str, Any]:
    """
    Query Gemini API usage from Cloud Monitoring.
    
    This queries the actual API request/token metrics.
    """
    if not HAS_CLOUD_LIBS:
        print("❌ google-cloud-monitoring not installed")
        print("   pip install google-cloud-monitoring")
        return {}
    
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"
    
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=days)
    
    # Time interval
    interval = monitoring_v3.TimeInterval({
        "start_time": {"seconds": int(start_time.timestamp())},
        "end_time": {"seconds": int(now.timestamp())},
    })
    
    results = {}
    
    # Metrics to query for Generative AI / Gemini API
    # These are the actual metric types used by Google AI Studio / Vertex AI
    metric_types = [
        # Google AI Studio (Generative Language API)
        "generativelanguage.googleapis.com/quota/generate_content_requests/usage",
        "generativelanguage.googleapis.com/quota/generate_content_input_token_count/usage",
        "generativelanguage.googleapis.com/quota/generate_content_output_token_count/usage",
        # Vertex AI Gemini
        "aiplatform.googleapis.com/prediction/online_prediction_count",
        "aiplatform.googleapis.com/prediction/online_prediction_tokens",
    ]
    
    for metric_type in metric_types:
        try:
            request = monitoring_v3.ListTimeSeriesRequest(
                name=project_name,
                filter=f'metric.type = "{metric_type}"',
                interval=interval,
                view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            )
            
            page_result = client.list_time_series(request=request)
            
            for time_series in page_result:
                metric_name = time_series.metric.type.split("/")[-1]
                labels = dict(time_series.metric.labels)
                
                total = 0
                for point in time_series.points:
                    total += point.value.int64_value or point.value.double_value or 0
                
                key = f"{metric_name}"
                if labels.get("model"):
                    key += f"_{labels['model']}"
                
                results[key] = {
                    "total": total,
                    "labels": labels,
                    "metric_type": metric_type,
                }
                
        except Exception as e:
            # Metric might not exist for this project
            pass
    
    return results


# =============================================================================
# Method 3: API Key Usage via Google AI Studio
# =============================================================================

def get_api_key_usage(api_key: str | None = None) -> dict[str, Any]:
    """
    Get usage for a specific API key via Google AI Studio.
    
    Note: This endpoint may have limited historical data.
    """
    import os
    
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ No API key provided")
        print("   Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        return {}
    
    # There's no official "usage" endpoint, but we can check quota
    # via a models.list call and inspect headers
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    try:
        response = requests.get(url)
        
        # Check rate limit headers for usage hints
        headers = dict(response.headers)
        usage_info = {
            "rate_limit_remaining": headers.get("x-ratelimit-remaining-requests"),
            "rate_limit_limit": headers.get("x-ratelimit-limit-requests"),
            "quota_user": headers.get("x-goog-quota-user"),
        }
        
        return {k: v for k, v in usage_info.items() if v is not None}
        
    except Exception as e:
        print(f"⚠️ Failed to query API: {e}")
        return {}


# =============================================================================
# Method 4: Cloud Billing Catalog (for pricing reference)
# =============================================================================

def get_gemini_pricing() -> list[dict]:
    """
    Get current Gemini API pricing from Cloud Billing Catalog.
    
    This fetches the actual pricing SKUs.
    """
    if not HAS_CLOUD_LIBS:
        print("❌ google-cloud-billing not installed")
        return []
    
    try:
        client = billing_v1.CloudCatalogClient()
        
        # List services to find Generative AI
        services = client.list_services()
        
        gemini_skus = []
        for service in services:
            if "generative" in service.display_name.lower() or "gemini" in service.display_name.lower():
                # Get SKUs for this service
                skus = client.list_skus(parent=service.name)
                for sku in skus:
                    if "gemini" in sku.description.lower() or "token" in sku.description.lower():
                        pricing = None
                        if sku.pricing_info:
                            pricing = {
                                "currency": sku.pricing_info[0].pricing_expression.tiered_rates[0].unit_price.currency_code if sku.pricing_info[0].pricing_expression.tiered_rates else None,
                                "nanos": sku.pricing_info[0].pricing_expression.tiered_rates[0].unit_price.nanos if sku.pricing_info[0].pricing_expression.tiered_rates else None,
                            }
                        
                        gemini_skus.append({
                            "sku_id": sku.sku_id,
                            "description": sku.description,
                            "service": service.display_name,
                            "pricing": pricing,
                        })
        
        return gemini_skus
        
    except Exception as e:
        print(f"⚠️ Failed to get pricing: {e}")
        return []


# =============================================================================
# Method 5: gcloud CLI wrapper
# =============================================================================

def get_usage_via_gcloud(days: int = 30) -> str:
    """
    Get usage via gcloud CLI commands.
    
    This is a fallback that doesn't require Python libraries.
    """
    import subprocess
    
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    end_date = datetime.now().strftime("%Y-%m-%dT23:59:59Z")
    
    # Query Cloud Monitoring for Gemini metrics
    cmd = [
        "gcloud", "monitoring", "metrics", "list",
        "--project", PROJECT_ID,
        "--filter", "metric.type:generativelanguage",
        "--format", "json",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except FileNotFoundError:
        return "gcloud CLI not found"


# =============================================================================
# Main report
# =============================================================================

def generate_usage_report(days: int = 30) -> None:
    """Generate a comprehensive usage report."""
    
    print("=" * 60)
    print("📊 GEMINI API USAGE REPORT")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Period: Last {days} days")
    print("=" * 60)
    print()
    
    # Try Cloud Monitoring
    print("🔍 Querying Cloud Monitoring API...")
    monitoring_data = get_usage_from_monitoring(days=days)
    
    if monitoring_data:
        print("\n📈 Cloud Monitoring Metrics:")
        print("-" * 40)
        for key, value in monitoring_data.items():
            print(f"  {key}: {value['total']:,}")
            if value.get('labels'):
                for k, v in value['labels'].items():
                    print(f"    └─ {k}: {v}")
    else:
        print("  ⚠️ No metrics found (may need to enable monitoring)")
    
    print()
    
    # Try BigQuery (if billing export is set up)
    print("🔍 Checking BigQuery billing export...")
    bq_data = get_usage_from_bigquery(
        start_date=(datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    )
    
    if bq_data:
        print("\n💰 Billing Data (from BigQuery):")
        print("-" * 40)
        total_cost = sum(row.get('cost', 0) for row in bq_data)
        print(f"  Total cost: ${total_cost:.2f}")
        
        # Group by SKU
        by_sku: dict[str, float] = {}
        for row in bq_data:
            sku = row.get('sku', 'unknown')
            by_sku[sku] = by_sku.get(sku, 0) + row.get('cost', 0)
        
        for sku, cost in sorted(by_sku.items(), key=lambda x: -x[1]):
            print(f"  {sku}: ${cost:.2f}")
    else:
        print("  ⚠️ No BigQuery data (billing export may not be set up)")
        print("  ℹ️  Set up billing export: https://cloud.google.com/billing/docs/how-to/export-data-bigquery")
    
    print()
    
    # Show pricing reference
    print("📋 Gemini API Pricing Reference:")
    print("-" * 40)
    for model, prices in PRICING.items():
        print(f"  {model}:")
        print(f"    Input:  ${prices['input_per_1m']}/1M tokens")
        print(f"    Output: ${prices['output_per_1m']}/1M tokens")
    
    print()
    print("=" * 60)
    print("💡 Tips for accurate tracking:")
    print("   1. Enable BigQuery billing export for actual costs")
    print("   2. Use Cloud Monitoring for real-time token counts")
    print("   3. Check AI Studio console: https://aistudio.google.com/")
    print("=" * 60)


# =============================================================================
# Quick script to query AI Studio console data
# =============================================================================

def get_aistudio_usage_instructions() -> str:
    """Instructions to get usage from AI Studio console."""
    return """
📊 To get usage data from Google AI Studio:

1. Go to https://aistudio.google.com/app/plan
2. You'll see:
   - Requests per day/minute
   - Token usage breakdown
   - Model-specific quotas

3. For detailed API key usage:
   - Go to https://console.cloud.google.com/apis/credentials
   - Click on your API key
   - View "Metrics" tab for request counts

4. For billing details:
   - Go to https://console.cloud.google.com/billing
   - Select your billing account
   - View "Reports" → Filter by "Generative Language API"

5. Export billing data to BigQuery:
   - Go to Billing → Billing export
   - Enable "Standard usage cost" export to BigQuery
   - Query the exported data for detailed analysis
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Gemini API usage from Google Cloud")
    parser.add_argument("--days", type=int, default=30, help="Number of days to query")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--project", type=str, default=PROJECT_ID, help="GCP Project ID")
    parser.add_argument("--instructions", action="store_true", help="Show instructions for AI Studio")
    
    args = parser.parse_args()
    
    if args.instructions:
        print(get_aistudio_usage_instructions())
    else:
        generate_usage_report(days=args.days)
