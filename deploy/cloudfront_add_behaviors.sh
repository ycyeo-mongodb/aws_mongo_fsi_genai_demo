#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Add two new behaviors + two new origins to the EXISTING CloudFront
# distribution `dk1wlzfbn3w8f.cloudfront.net`:
#
#   1. PathPattern  /fsi_digital_bank_demo/api/*
#      Origin       <asean_sa_yc>.execute-api.<region>.amazonaws.com
#      Behavior     All methods · forward Host header off · TLS only
#
#   2. PathPattern  /fsi_digital_bank_demo/*
#      Origin       <s3-bucket>.s3.<region>.amazonaws.com (via OAC)
#      Behavior     GET/HEAD only · cached
#
# ⚠️  This script MUTATES a CloudFront distribution that may be serving
#     other production traffic. It will:
#       1. Print exactly what it intends to add
#       2. Save a JSON backup of the current config to deploy/cf-backup-<ts>.json
#       3. ASK for confirmation before calling update-distribution
#
# Run with:   ./cloudfront_add_behaviors.sh
# Dry-run:    DRY_RUN=1 ./cloudfront_add_behaviors.sh
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/deploy.env"

: "${AWS_REGION:?AWS_REGION must be set in deploy.env}"
: "${ASEAN_SA_YC_API_ID:?ASEAN_SA_YC_API_ID must be set}"
: "${CLOUDFRONT_DISTRIBUTION_ID:?CLOUDFRONT_DISTRIBUTION_ID must be set}"

STACK_NAME="${STACK_NAME:-${PROJECT_NAME:-fsi-digital-bank-demo}}"
PATH_PREFIX="${PATH_PREFIX:-/fsi_digital_bank_demo}"
# CloudFront path patterns can't start with a leading slash; strip it.
CF_PATH_PREFIX="${PATH_PREFIX#/}"

API_ORIGIN_DOMAIN="${ASEAN_SA_YC_API_ID}.execute-api.${AWS_REGION}.amazonaws.com"

S3_BUCKET="$(aws cloudformation describe-stacks \
  --region "${AWS_REGION}" \
  --stack-name "${STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='StaticBucketName'].OutputValue" \
  --output text)"
S3_ORIGIN_DOMAIN="${S3_BUCKET}.s3.${AWS_REGION}.amazonaws.com"

API_ORIGIN_ID="ApiGw-${ASEAN_SA_YC_API_ID}"
S3_ORIGIN_ID="S3-${S3_BUCKET}"

# ── Backup current config ────────────────────────────────────────────────
TS="$(date +%Y%m%d-%H%M%S)"
BACKUP_FILE="${SCRIPT_DIR}/cf-backup-${TS}.json"
WORK_FILE="${SCRIPT_DIR}/cf-work-${TS}.json"

aws cloudfront get-distribution-config \
  --id "${CLOUDFRONT_DISTRIBUTION_ID}" \
  --output json > "${BACKUP_FILE}"
ETAG="$(jq -r '.ETag' "${BACKUP_FILE}")"
jq '.DistributionConfig' "${BACKUP_FILE}" > "${WORK_FILE}"

echo "✓ Backup of current config: ${BACKUP_FILE}"
echo "ETag for update: ${ETAG}"

# ── Idempotency check: bail out if a behavior with our path pattern exists ─
EXISTING="$(jq -r --arg p "${CF_PATH_PREFIX}/*" \
  '.CacheBehaviors.Items[]? | select(.PathPattern == $p) | .PathPattern' \
  "${WORK_FILE}" || true)"
if [[ -n "${EXISTING}" ]]; then
  echo "ℹ️  A behavior for '${CF_PATH_PREFIX}/*' already exists on this distribution."
  echo "Nothing to do. (To re-apply, delete the existing behavior first.)"
  exit 0
fi

# ── Look up / create an Origin Access Control for the S3 origin ─────────
echo "Looking up CloudFront OAC for the S3 origin..."
OAC_NAME="${PROJECT_NAME:-fsi-digital-bank-demo}-oac"
OAC_ID="$(aws cloudfront list-origin-access-controls \
  --query "OriginAccessControlList.Items[?Name=='${OAC_NAME}'].Id | [0]" \
  --output text)"
if [[ -z "${OAC_ID}" || "${OAC_ID}" == "None" ]]; then
  echo "Creating OAC '${OAC_NAME}'..."
  OAC_ID="$(aws cloudfront create-origin-access-control \
    --origin-access-control-config "{
      \"Name\": \"${OAC_NAME}\",
      \"OriginAccessControlOriginType\": \"s3\",
      \"SigningBehavior\": \"always\",
      \"SigningProtocol\": \"sigv4\"
    }" \
    --query "OriginAccessControl.Id" \
    --output text)"
fi
echo "Using OAC: ${OAC_ID}"

# ── Patch the config: add 2 origins + 2 cache behaviors ─────────────────
# We use AllViewer cache policy and AllViewerExceptHostHeader origin request
# policy (these are AWS-managed and exist by default).
CACHE_DISABLED_POLICY="4135ea2d-6df8-44a3-9df3-4b5a84be39ad"   # Managed-CachingDisabled
CACHE_OPTIMIZED_POLICY="658327ea-f89d-4fab-a63d-7e88639e58f6"  # Managed-CachingOptimized
ORIGIN_REQ_ALL_VIEWER_EXCEPT_HOST="b689b0a8-53d0-40ab-baf2-68738e2966ac"  # Managed-AllViewerExceptHostHeader

# Build the new origins as JSON snippets and inject them with jq.
jq \
  --arg api_id "${API_ORIGIN_ID}" \
  --arg api_domain "${API_ORIGIN_DOMAIN}" \
  --arg s3_id "${S3_ORIGIN_ID}" \
  --arg s3_domain "${S3_ORIGIN_DOMAIN}" \
  --arg oac_id "${OAC_ID}" \
  --arg path_pat_api "${CF_PATH_PREFIX}/api/*" \
  --arg path_pat_root "${CF_PATH_PREFIX}/*" \
  --arg cache_disabled "${CACHE_DISABLED_POLICY}" \
  --arg cache_optimized "${CACHE_OPTIMIZED_POLICY}" \
  --arg origin_req "${ORIGIN_REQ_ALL_VIEWER_EXCEPT_HOST}" \
  '
  # ── 1. Add the API Gateway origin if missing ──
  .Origins.Items += (
    if [.Origins.Items[].Id] | index($api_id) then []
    else [{
      Id: $api_id,
      DomainName: $api_domain,
      OriginPath: "",
      CustomHeaders: { Quantity: 0 },
      CustomOriginConfig: {
        HTTPPort: 80, HTTPSPort: 443,
        OriginProtocolPolicy: "https-only",
        OriginSslProtocols: { Quantity: 1, Items: ["TLSv1.2"] },
        OriginReadTimeout: 30, OriginKeepaliveTimeout: 5
      },
      ConnectionAttempts: 3, ConnectionTimeout: 10,
      OriginShield: { Enabled: false }
    }] end
  ) |
  .Origins.Quantity = (.Origins.Items | length) |

  # ── 2. Add the S3 origin if missing ──
  .Origins.Items += (
    if [.Origins.Items[].Id] | index($s3_id) then []
    else [{
      Id: $s3_id,
      DomainName: $s3_domain,
      OriginPath: "",
      CustomHeaders: { Quantity: 0 },
      S3OriginConfig: { OriginAccessIdentity: "" },
      OriginAccessControlId: $oac_id,
      ConnectionAttempts: 3, ConnectionTimeout: 10,
      OriginShield: { Enabled: false }
    }] end
  ) |
  .Origins.Quantity = (.Origins.Items | length) |

  # ── 3. Prepend the API behavior (MORE specific path first) ──
  .CacheBehaviors.Items = ([
    {
      PathPattern: $path_pat_api,
      TargetOriginId: $api_id,
      ViewerProtocolPolicy: "redirect-to-https",
      AllowedMethods: {
        Quantity: 7,
        Items: ["GET","HEAD","OPTIONS","PUT","POST","PATCH","DELETE"],
        CachedMethods: { Quantity: 2, Items: ["GET","HEAD"] }
      },
      Compress: true,
      CachePolicyId: $cache_disabled,
      OriginRequestPolicyId: $origin_req,
      SmoothStreaming: false,
      FieldLevelEncryptionId: "",
      LambdaFunctionAssociations: { Quantity: 0 },
      FunctionAssociations: { Quantity: 0 }
    },
    {
      PathPattern: $path_pat_root,
      TargetOriginId: $s3_id,
      ViewerProtocolPolicy: "redirect-to-https",
      AllowedMethods: {
        Quantity: 2,
        Items: ["GET","HEAD"],
        CachedMethods: { Quantity: 2, Items: ["GET","HEAD"] }
      },
      Compress: true,
      CachePolicyId: $cache_optimized,
      SmoothStreaming: false,
      FieldLevelEncryptionId: "",
      LambdaFunctionAssociations: { Quantity: 0 },
      FunctionAssociations: { Quantity: 0 }
    }
  ] + (.CacheBehaviors.Items // [])) |
  .CacheBehaviors.Quantity = (.CacheBehaviors.Items | length)
  ' "${WORK_FILE}" > "${WORK_FILE}.patched"

echo
echo "─── Plan ──────────────────────────────────────────────────────────"
echo "Adding origin:  ${API_ORIGIN_ID} → ${API_ORIGIN_DOMAIN}"
echo "Adding origin:  ${S3_ORIGIN_ID}  → ${S3_ORIGIN_DOMAIN}   (OAC ${OAC_ID})"
echo "Adding behavior: ${CF_PATH_PREFIX}/api/*  → ${API_ORIGIN_ID}   (CachingDisabled)"
echo "Adding behavior: ${CF_PATH_PREFIX}/*      → ${S3_ORIGIN_ID}    (CachingOptimized)"
echo "Default behavior left untouched."
echo "Diff (current vs patched):"
diff <(jq -S . "${WORK_FILE}") <(jq -S . "${WORK_FILE}.patched") | head -80 || true
echo "──────────────────────────────────────────────────────────────────"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — not applying. Patched config saved at ${WORK_FILE}.patched"
  exit 0
fi

read -r -p "Apply these changes to distribution ${CLOUDFRONT_DISTRIBUTION_ID}? [y/N] " REPLY
if [[ ! "${REPLY}" =~ ^[Yy]$ ]]; then
  echo "Aborted. Backup is at ${BACKUP_FILE} if you need it."
  exit 1
fi

aws cloudfront update-distribution \
  --id "${CLOUDFRONT_DISTRIBUTION_ID}" \
  --if-match "${ETAG}" \
  --distribution-config "file://${WORK_FILE}.patched" \
  --output table \
  --query "Distribution.{Id:Id,Status:Status,DomainName:DomainName,LastModifiedTime:LastModifiedTime}"

# ── Wait + S3 bucket policy tightening ──────────────────────────────────
echo
echo "✓ CloudFront update requested. Propagation usually 3-5 min."
echo "  Track: aws cloudfront wait distribution-deployed --id ${CLOUDFRONT_DISTRIBUTION_ID}"

# Tighten the S3 bucket policy now that we know which distribution will use it.
DIST_ARN="arn:aws:cloudfront::$(aws sts get-caller-identity --query Account --output text):distribution/${CLOUDFRONT_DISTRIBUTION_ID}"
cat > "${SCRIPT_DIR}/s3-policy-tight.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "AllowCloudFrontServicePrincipalReadOnly",
    "Effect": "Allow",
    "Principal": { "Service": "cloudfront.amazonaws.com" },
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::${S3_BUCKET}/*",
    "Condition": {
      "StringEquals": {
        "AWS:SourceArn": "${DIST_ARN}"
      }
    }
  }]
}
EOF
aws s3api put-bucket-policy --bucket "${S3_BUCKET}" --policy "file://${SCRIPT_DIR}/s3-policy-tight.json"
echo "✓ Tightened S3 bucket policy to only allow distribution ${CLOUDFRONT_DISTRIBUTION_ID}."

echo
echo "Open in browser once the distribution is Deployed:"
echo "  https://dk1wlzfbn3w8f.cloudfront.net${PATH_PREFIX}/"
