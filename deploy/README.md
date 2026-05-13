# Deploying the FSI Digital Bank demo to AWS

End state:

```
[ Audience ]
    │
    ▼
dk1wlzfbn3w8f.cloudfront.net
    │
    ├── /fsi_digital_bank_demo/*       →  S3 bucket (static index.html)
    └── /fsi_digital_bank_demo/api/*   →  asean_sa_yc HTTP API
                                              │
                                              ▼
                                       Lambda container (FastAPI via Mangum)
                                              │
                                              ▼
                          MongoDB Atlas · Voyage AI · LLM API Gateway
```

## What's in this folder

| File | Purpose |
|---|---|
| `Dockerfile` | Lambda container image (Python 3.12, FastAPI + Mangum) |
| `../lambda_handler.py` | Mangum wrapper that strips the `/fsi_digital_bank_demo` prefix |
| `build_static.sh` | Rewrites `/api/*` → `/fsi_digital_bank_demo/api/*` in `index.html` |
| `cloudformation.yaml` | Lambda + IAM role + log group + S3 bucket + 2 routes on `asean_sa_yc` |
| `deploy.env.example` | Config template — copy to `deploy.env` and fill in |
| `deploy.sh` | One-shot orchestrator (build → push ECR → deploy CFN → sync S3 → smoke test) |
| `cloudfront_add_behaviors.sh` | Adds 2 origins + 2 behaviors to the **existing** distribution (interactive, with confirmation) |

Files generated at runtime (gitignored):
- `deploy.env` (your filled-in secrets)
- `cf-backup-<timestamp>.json` (saved before any CloudFront mutation)
- `cf-work-<timestamp>.json[.patched]` (the diff applied to CloudFront)
- `s3-policy-tight.json` (the tightened bucket policy)
- `../build/static/index.html` (the rewritten static file)

## Prerequisites

On your workstation:

```bash
aws --version          # ≥ 2.x
docker --version       # daemon running
jq --version
```

In AWS:
1. **`asean_sa_yc` HTTP API exists** — get its ApiId:
   ```bash
   aws apigatewayv2 get-apis \
     --query "Items[?Name=='asean_sa_yc'].{Id:ApiId,Endpoint:ApiEndpoint}" --output table
   ```
2. **CloudFront distribution `dk1wlzfbn3w8f.cloudfront.net` exists** — get its Id:
   ```bash
   aws cloudfront list-distributions \
     --query "DistributionList.Items[?DomainName=='dk1wlzfbn3w8f.cloudfront.net'].Id" \
     --output text
   ```
3. **Caller permissions** — your AWS identity needs:
   - `iam:CreateRole`, `iam:AttachRolePolicy`, `iam:PutRolePolicy`, `iam:PassRole`
   - `lambda:*` on `fsi-digital-bank-demo-api`
   - `ecr:CreateRepository`, `ecr:*` on `fsi-digital-bank-demo`
   - `apigateway:*` on the `asean_sa_yc` API
   - `s3:CreateBucket`, `s3:Put*` on the new bucket
   - `cloudformation:*`
   - `cloudfront:GetDistributionConfig`, `cloudfront:UpdateDistribution`, `cloudfront:Create/ListOriginAccessControl`

## Step-by-step

### 1. Fill in `deploy.env`

```bash
cd deploy
cp deploy.env.example deploy.env
$EDITOR deploy.env
```

Required values: `AWS_REGION`, `ASEAN_SA_YC_API_ID`, `MONGODB_URI`, `VOYAGE_API_KEY`, `LLM_API_URL`.
For step 3 you'll also need `CLOUDFRONT_DISTRIBUTION_ID`.

### 2. Run the main deploy

```bash
./deploy.sh
```

Roughly what it does (with timings on a typical first run):

| Phase | What happens | First-run time |
|---|---|---|
| 0 | `aws sts get-caller-identity` sanity check | <1 s |
| 1 | `build_static.sh` rewrites `index.html` | <1 s |
| 2 | `docker build --platform linux/amd64` for the Lambda image | ~90 s |
| 3 | Create ECR repo if missing, `docker push` | ~60 s (depends on image size + upload speed) |
| 4 | `aws cloudformation deploy` creates Lambda + role + S3 + API GW routes | ~60 s |
| 5 | `lambda update-function-code` + wait for status `Successful` | ~20 s |
| 6 | `aws s3 sync` the rewritten `index.html` | <5 s |
| 7 | `curl` test against the API GW URL | ~5 s |

When it finishes you'll get an output like:
```
✅ Deploy complete.
Stack:              fsi-digital-bank-demo
Image:              979559056307.dkr.ecr.us-east-1.amazonaws.com/fsi-digital-bank-demo:20260513-145002
S3 bucket:          fsi-digital-bank-demo-static-979559056307-us-east-1
API GW test URL:    https://abcd1234.execute-api.us-east-1.amazonaws.com/fsi_digital_bank_demo/api/customers
```

You can hit that API GW URL directly to confirm the backend works *before* touching CloudFront.

### 3. Wire it onto your existing CloudFront

⚠️ **This script modifies the existing `dk1wlzfbn3w8f.cloudfront.net` distribution.** It:

1. Saves the current `DistributionConfig` to `cf-backup-<timestamp>.json` (rollback safety).
2. Computes the new config (adds 2 origins + 2 cache behaviors).
3. **Prints a diff and prompts for confirmation** before calling `update-distribution`.
4. After the update succeeds, tightens the S3 bucket policy to allow only this specific distribution.

```bash
# DRY RUN first (no changes — just prints the diff)
DRY_RUN=1 ./cloudfront_add_behaviors.sh

# Real run
./cloudfront_add_behaviors.sh
```

CloudFront propagation typically takes 3–5 minutes. You can track it:

```bash
aws cloudfront wait distribution-deployed --id $CLOUDFRONT_DISTRIBUTION_ID
```

Once deployed:

```
https://dk1wlzfbn3w8f.cloudfront.net/fsi_digital_bank_demo/
```

### 4. MongoDB Atlas IP allowlist

By default Lambdas come from AWS-owned IPs that change. For a workshop demo, the quickest path is to **add `0.0.0.0/0` to the Atlas Network Access list**. For anything production-bound, either:

- Put the Lambda inside a VPC with a NAT gateway and allowlist its EIP, or
- Use **MongoDB Atlas PrivateLink** (Atlas → Network Access → Private Endpoint → AWS PrivateLink).

The IAM role created by CFN gives Lambda **basic execution** and **CloudWatch Logs only** — it doesn't talk to Bedrock directly (the app calls `LLM_API_URL`, which is itself an API Gateway). If you ever switch the app to call `bedrock-runtime:InvokeModel` directly, add that permission to `LambdaExecutionRole`.

## Rolling back

```bash
# Backend (Lambda + S3 + routes)
aws cloudformation delete-stack \
  --stack-name fsi-digital-bank-demo \
  --region $AWS_REGION

# CloudFront (revert to the saved backup)
ETAG=$(jq -r .ETag deploy/cf-backup-<timestamp>.json)
jq .DistributionConfig deploy/cf-backup-<timestamp>.json > deploy/cf-rollback.json
aws cloudfront update-distribution \
  --id $CLOUDFRONT_DISTRIBUTION_ID \
  --if-match $ETAG \
  --distribution-config file://deploy/cf-rollback.json
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `curl` to API GW returns 502 | Lambda cold start timed out, or Mongo unreachable | `aws logs tail /aws/lambda/fsi-digital-bank-demo-api --follow` |
| HTTP 404 from CloudFront | Behavior order is wrong — the default catches first | Make sure the `/fsi_digital_bank_demo/api/*` behavior is listed BEFORE the `/fsi_digital_bank_demo/*` behavior |
| HTTP 403 from S3 origin | Bucket policy doesn't allow this distribution | Re-run `cloudfront_add_behaviors.sh` — the last step writes the tightened policy |
| Lambda hits 30s timeout on FAQ/credit-score | The default LangGraph investigator takes 9–11 s; first call may be a cold start | Bump `LambdaMemoryMb` to 1536 in `deploy.env`, or warm the function with `aws lambda invoke` before the demo |
| `docker push` fails with `no basic auth credentials` | The ECR login expired | Re-run `./deploy.sh` — it logs in fresh each time |

## What this stack does NOT do

- Modify your existing CloudFront distribution **without confirmation**.
- Store secrets in plain text on disk — `deploy.env` is gitignored and Lambda env vars are encrypted at rest (KMS aws/lambda).
- Take any action against MongoDB Atlas — IP allowlist changes are manual.
- Touch any resources outside the names prefixed with `fsi-digital-bank-demo-*`.
