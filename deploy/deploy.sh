#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# FSI Digital Bank demo — one-shot deploy orchestrator.
#
# WHAT THIS DOES
#   1. Builds the production index.html with the CloudFront path prefix
#   2. Builds the Lambda container image
#   3. Pushes the image to ECR (creates the repo on first run)
#   4. Deploys the CloudFormation stack (creates Lambda, IAM role, S3 bucket,
#      attaches routes to the existing asean_sa_yc API Gateway)
#   5. Uploads the built index.html to S3
#
# WHAT THIS DOES NOT DO
#   - Modify the existing CloudFront distribution. Run
#     `cloudfront_add_behaviors.sh` separately for that — it lists what it
#     plans to change before applying.
#
# PREREQUISITES
#   - AWS credentials in the environment (or a working profile via --profile).
#     This script will FAIL if you only have temporary credentials about to
#     expire. Run `aws sts get-caller-identity` first to verify.
#   - Docker daemon running locally.
#   - jq, aws CLI v2.
#
# USAGE
#   Copy deploy.env.example to deploy.env, fill it in, then:
#       cd deploy && ./deploy.sh
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${SCRIPT_DIR}/deploy.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  echo "Copy deploy.env.example to deploy.env and fill in the values." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${AWS_REGION:?AWS_REGION must be set in deploy.env}"
: "${ASEAN_SA_YC_API_ID:?ASEAN_SA_YC_API_ID must be set in deploy.env}"
: "${MONGODB_URI:?MONGODB_URI must be set in deploy.env}"
: "${VOYAGE_API_KEY:?VOYAGE_API_KEY must be set in deploy.env}"
: "${LLM_API_URL:?LLM_API_URL must be set in deploy.env}"

PROJECT_NAME="${PROJECT_NAME:-fsi-digital-bank-demo}"
PATH_PREFIX="${PATH_PREFIX:-/fsi_digital_bank_demo}"
STACK_NAME="${STACK_NAME:-${PROJECT_NAME}}"
ECR_REPO_NAME="${ECR_REPO_NAME:-${PROJECT_NAME}}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"

bold(){ printf '\033[1m%s\033[0m\n' "$*"; }
section(){ printf '\n\033[1;36m─── %s ───\033[0m\n' "$*"; }

# ── 0. Identity sanity check ─────────────────────────────────────────────
section "0. AWS identity check"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
CALLER_ARN="$(aws sts get-caller-identity --query Arn --output text)"
echo "Account: ${ACCOUNT_ID}"
echo "Caller:  ${CALLER_ARN}"
echo "Region:  ${AWS_REGION}"

ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_IMAGE_URI="${ECR_REGISTRY}/${ECR_REPO_NAME}:${IMAGE_TAG}"
ECR_IMAGE_URI_LATEST="${ECR_REGISTRY}/${ECR_REPO_NAME}:latest"

# ── 1. Build the production index.html ──────────────────────────────────
section "1. Build production index.html"
PATH_PREFIX="${PATH_PREFIX}" "${SCRIPT_DIR}/build_static.sh"

# ── 2. Build the Lambda container image ─────────────────────────────────
section "2. Build Lambda container image"
# Lambda Python images run on x86_64; build accordingly even on Apple Silicon.
docker build \
  --platform linux/amd64 \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${PROJECT_NAME}:${IMAGE_TAG}" \
  -t "${PROJECT_NAME}:latest" \
  "${REPO_ROOT}"

# ── 3. Create ECR repo if missing, login, push ─────────────────────────
section "3. Push image to ECR"
if ! aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "Creating ECR repo ${ECR_REPO_NAME} ..."
  aws ecr create-repository \
    --repository-name "${ECR_REPO_NAME}" \
    --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE \
    >/dev/null
fi

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

docker tag "${PROJECT_NAME}:${IMAGE_TAG}" "${ECR_IMAGE_URI}"
docker tag "${PROJECT_NAME}:latest"       "${ECR_IMAGE_URI_LATEST}"
docker push "${ECR_IMAGE_URI}"
docker push "${ECR_IMAGE_URI_LATEST}"
echo "Pushed: ${ECR_IMAGE_URI}"

# ── 4. Deploy / update CloudFormation stack ─────────────────────────────
section "4. Deploy CloudFormation stack"
aws cloudformation deploy \
  --region "${AWS_REGION}" \
  --stack-name "${STACK_NAME}" \
  --template-file "${SCRIPT_DIR}/cloudformation.yaml" \
  --capabilities CAPABILITY_NAMED_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides \
    ProjectName="${PROJECT_NAME}" \
    PathPrefix="${PATH_PREFIX}" \
    AseanSaYcApiId="${ASEAN_SA_YC_API_ID}" \
    EcrImageUri="${ECR_IMAGE_URI}" \
    MongoDbUri="${MONGODB_URI}" \
    VoyageApiKey="${VOYAGE_API_KEY}" \
    LlmApiUrl="${LLM_API_URL}"

# ── 5. Push image update onto Lambda explicitly ─────────────────────────
# CloudFormation only re-deploys the Lambda when EcrImageUri changes. Since
# we tag with a timestamp, that's always the case here — but we still call
# update-function-code in case ImmutableTag policy bites us.
section "5. Update Lambda function code"
aws lambda update-function-code \
  --region "${AWS_REGION}" \
  --function-name "${PROJECT_NAME}-api" \
  --image-uri "${ECR_IMAGE_URI}" \
  --publish \
  --output text \
  --query 'LastUpdateStatus' || true

# Wait for the update to settle so the smoke test below hits the new image.
aws lambda wait function-updated \
  --region "${AWS_REGION}" \
  --function-name "${PROJECT_NAME}-api"

# ── 6. Sync built static index.html to S3 ───────────────────────────────
section "6. Sync static site to S3"
BUCKET="$(aws cloudformation describe-stacks \
  --region "${AWS_REGION}" \
  --stack-name "${STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='StaticBucketName'].OutputValue" \
  --output text)"
echo "Uploading to s3://${BUCKET}/"

aws s3 sync "${REPO_ROOT}/build/static/" "s3://${BUCKET}/" \
  --region "${AWS_REGION}" \
  --delete \
  --cache-control "public, max-age=60" \
  --content-type "text/html; charset=utf-8" \
  --exclude "*" --include "*.html"

# ── 7. Smoke test the deployed Lambda via API Gateway ───────────────────
section "7. Smoke test"
API_INVOKE_URL="$(aws cloudformation describe-stacks \
  --region "${AWS_REGION}" \
  --stack-name "${STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='ApiInvokeUrlExample'].OutputValue" \
  --output text)"
echo "Hitting ${API_INVOKE_URL} ..."
HTTP_CODE=$(curl -s -o /tmp/fsi_deploy_test.json -w '%{http_code}' --max-time 30 "${API_INVOKE_URL}")
echo "HTTP ${HTTP_CODE}"
echo "Body preview:"
head -c 400 /tmp/fsi_deploy_test.json || true
echo

# ── Summary ─────────────────────────────────────────────────────────────
bold ""
bold "✅ Deploy complete."
bold ""
echo "Stack:              ${STACK_NAME}"
echo "Image:              ${ECR_IMAGE_URI}"
echo "S3 bucket:          ${BUCKET}"
echo "API GW test URL:    ${API_INVOKE_URL}"
echo ""
echo "Next step → run cloudfront_add_behaviors.sh to add the two new"
echo "behaviors to your existing dk1wlzfbn3w8f.cloudfront.net distribution."
