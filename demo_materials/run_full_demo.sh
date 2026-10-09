#!/usr/bin/env bash
# 一键体验理赔 Agent 全链路：真实状态机 + 合成材料，无需准备任何真实数据。
# 用法：./demo_materials/run_full_demo.sh [base_url]   默认 http://127.0.0.1:8001
# 每次运行自动生成新的案件号，可重复执行，不会撞“案件已存在”。
set -euo pipefail
BASE="${1:-http://127.0.0.1:8001}"
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$DIR")"
RUN="$(date +%H%M%S)"
CLAIM_A="CLM-2026-001-$RUN"   # 完整材料案件
CLAIM_B="CLM-2026-002-$RUN"   # 缺保单号案件

# 基于模板生成当次案件号的 JSON（用法: payload 模板.json 新案件号）
payload() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); d["claim_id"]=sys.argv[2]; print(json.dumps(d,ensure_ascii=False))' "$1" "$2"; }

# 从 .env 读取密钥（本地演示用）
API_KEY="$(grep -E '^CLAIMS_API_KEY=' "$ROOT/.env" | cut -d= -f2- | tr -d '"')"
REVIEWER_KEY="$(grep -E '^REVIEWER_API_KEY=' "$ROOT/.env" | cut -d= -f2- | tr -d '"')"

hr() { printf '\n\033[1;36m══════ %s ══════\033[0m\n' "$1"; }
show() { python3 -m json.tool; }

hr "0. 健康检查"
curl -sf "$BASE/health" | show

hr "1. 演示一键全链路（固定合成案件，Web 页面「载入演示」同款）"
curl -sf -X POST "$BASE/api/demo/claims/process" | show

hr "2. 缺材料报案（无保单号 → 状态机中断等待补材料）"
curl -sf -X POST "$BASE/api/claims/process" \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d "$(payload "$DIR/02_missing_policy_claim.json" "$CLAIM_B")" | show

hr "2b. 补齐保单号（interrupt 恢复，案件继续流转）"
curl -sf -X POST "$BASE/api/claims/$CLAIM_B/supplement" \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d @"$DIR/03_supplement_policy.json" | show

hr "3. 材料齐全的完整报案（三专家并行 → 置信度融合 → 决策）"
curl -sf -X POST "$BASE/api/claims/process" \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d "$(payload "$DIR/01_full_flow_claim.json" "$CLAIM_A")" | show

hr "4. 人工审核（审核员密钥放行 awaiting_review 案件）"
curl -s -X POST "$BASE/api/claims/$CLAIM_A/review" \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" -H "X-Reviewer-Key: $REVIEWER_KEY" \
  -d @"$DIR/04_review_payload.json" | show

hr "5. 查询案件状态（LangGraph 检查点持久化，服务重启也不丢）"
curl -sf "$BASE/api/claims/$CLAIM_A" -H "X-API-Key: $API_KEY" | show

hr "6. Prometheus 业务指标"
curl -sf "$BASE/metrics" | grep -E '^claims_' | head -20

printf '\n\033[1;32m✔ 全链路体验完成。浏览器打开 %s 可在 Web 工作台重放以上流程。\033[0m\n' "$BASE"
