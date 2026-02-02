#!/bin/bash
# Setup BigQuery Billing Export for Gemini Usage Tracking

PROJECT_ID="gen-lang-client-0202182328"
DATASET_ID="billing_export"
BILLING_ACCOUNT="01D2F4-3F2B44-2B0851"

echo "Step 1: Create BigQuery dataset for billing export..."
bq --project_id=$PROJECT_ID mk --dataset --location=EU $DATASET_ID 2>/dev/null || echo "Dataset already exists"

echo ""
echo "Step 2: Enable billing export in Cloud Console"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Go to: https://console.cloud.google.com/billing/$BILLING_ACCOUNT/export"
echo ""
echo "Configure:"
echo "  • Standard usage cost → Enable"
echo "  • Project: $PROJECT_ID"
echo "  • Dataset: $DATASET_ID"
echo ""
echo "⏱️  Note: Data takes 24-48 hours to appear after enabling"
