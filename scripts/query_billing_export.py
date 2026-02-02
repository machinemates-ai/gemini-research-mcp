#!/usr/bin/env python3
"""
Query BigQuery Billing Export for Gemini API usage breakdown.
Shows input vs output tokens and actual costs.
"""

from google.cloud import bigquery
from datetime import datetime, timedelta
import sys

PROJECT_ID = "gen-lang-client-0202182328"
DATASET_ID = "billing_export"

# The billing export table name follows this pattern
# You may need to adjust based on your actual table name
TABLE_PATTERN = f"{PROJECT_ID}.{DATASET_ID}.gcp_billing_export_v1_*"


def query_gemini_costs(days: int = 30) -> None:
    """Query Gemini API costs from BigQuery billing export."""
    client = bigquery.Client(project=PROJECT_ID)
    
    # First, find the actual table name
    print(f"🔍 Looking for billing export tables in {DATASET_ID}...")
    
    tables_query = f"""
    SELECT table_name 
    FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.TABLES`
    WHERE table_name LIKE 'gcp_billing_export%'
    ORDER BY table_name DESC
    LIMIT 5
    """
    
    try:
        tables = list(client.query(tables_query).result())
        if not tables:
            print("\n⚠️  No billing export tables found yet.")
            print("   This usually means:")
            print("   1. Billing export not enabled yet")
            print("   2. Data hasn't been exported yet (takes 24-48 hours)")
            print(f"\n   Enable at: https://console.cloud.google.com/billing/01D2F4-3F2B44-2B0851/export")
            print(f"   Select project: {PROJECT_ID}")
            print(f"   Select dataset: {DATASET_ID}")
            return
            
        print(f"   Found tables: {[t.table_name for t in tables]}")
        table_name = tables[0].table_name
        
    except Exception as e:
        if "Not found" in str(e):
            print(f"\n⚠️  Dataset {DATASET_ID} not found or no access.")
            print("   Make sure billing export is configured.")
        else:
            print(f"\n❌ Error checking tables: {e}")
        return

    # Query for Gemini API costs with input/output breakdown
    print(f"\n📊 Querying Gemini costs for last {days} days...")
    
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
        -- Extract model info from labels if available
        (SELECT value FROM UNNEST(labels) WHERE key = 'model') as model
    FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`
    WHERE 
        service.description LIKE '%Generative%' 
        OR service.description LIKE '%Gemini%'
        OR service.description LIKE '%AI Platform%'
        OR service.description LIKE '%Vertex%'
    AND DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
    ORDER BY usage_start_time DESC
    LIMIT 1000
    """
    
    try:
        results = list(client.query(query).result())
        
        if not results:
            print("   No Gemini usage found in the specified period.")
            print("   (Data may still be processing)")
            return
            
        # Aggregate by SKU
        sku_totals = {}
        for row in results:
            sku = row.sku or "Unknown"
            if sku not in sku_totals:
                sku_totals[sku] = {"cost": 0, "usage": 0, "unit": row.usage_unit}
            sku_totals[sku]["cost"] += float(row.cost or 0)
            sku_totals[sku]["usage"] += float(row.usage_amount or 0)
        
        print("\n" + "=" * 70)
        print("GEMINI API USAGE BREAKDOWN")
        print("=" * 70)
        
        total_cost = 0
        input_cost = 0
        output_cost = 0
        
        for sku, data in sorted(sku_totals.items(), key=lambda x: -x[1]["cost"]):
            cost = data["cost"]
            total_cost += cost
            
            # Categorize as input or output
            sku_lower = sku.lower()
            if "input" in sku_lower or "prompt" in sku_lower:
                input_cost += cost
                category = "📥 INPUT"
            elif "output" in sku_lower or "response" in sku_lower or "completion" in sku_lower:
                output_cost += cost
                category = "📤 OUTPUT"
            else:
                category = "❓ OTHER"
            
            usage_str = f"{data['usage']:,.0f} {data['unit']}" if data['usage'] else "N/A"
            print(f"\n{category}")
            print(f"  SKU: {sku}")
            print(f"  Usage: {usage_str}")
            print(f"  Cost: ${cost:.4f}")
        
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"  📥 Input tokens cost:  ${input_cost:.2f}")
        print(f"  📤 Output tokens cost: ${output_cost:.2f}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  💰 Total Gemini cost:  ${total_cost:.2f}")
        
        if input_cost + output_cost > 0:
            input_pct = (input_cost / (input_cost + output_cost)) * 100
            output_pct = (output_cost / (input_cost + output_cost)) * 100
            print(f"\n  Ratio: {input_pct:.1f}% input / {output_pct:.1f}% output")
            
    except Exception as e:
        print(f"\n❌ Query error: {e}")
        if "does not exist" in str(e).lower():
            print("   The billing export table structure may be different.")
            print("   Check the BigQuery console to see actual table schema.")


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    query_gemini_costs(days)


if __name__ == "__main__":
    main()
