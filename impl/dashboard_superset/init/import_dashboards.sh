#!/usr/bin/env sh
# =============================================================================
# import_dashboards.sh
# First-time Superset initialisation: DB upgrade → admin user → datasources
# → dashboard import.
#
# Runs inside the `superset-init` Docker container defined in docker-compose.yml.
# Environment variables (injected by Docker Compose):
#   SUPERSET_BASE_URL        http://superset:8088  (or http://localhost:8088)
#   ADMIN_USERNAME           admin
#   ADMIN_PASSWORD           admin
#   ADMIN_EMAIL              admin@example.com
#   DATABASE_URL             postgresql+psycopg2://superset:superset@postgres:5432/superset
#   APP_DB_URL               postgresql+psycopg2://app_ro:app_ro@postgres:5432/maintenance
#   SUPERSET_CONFIG_PATH     /app/pythonpath/superset_config.py
# =============================================================================

set -eu

SUPERSET_URL="${SUPERSET_BASE_URL:-http://superset:8088}"
ADMIN_USER="${ADMIN_USERNAME:-admin}"
ADMIN_PASS="${ADMIN_PASSWORD:-admin}"
ADMIN_MAIL="${ADMIN_EMAIL:-admin@example.com}"
DASHBOARDS_DIR="/app/dashboards"
DATASOURCES_DIR="/app/datasources"

echo "============================================================"
echo " Superset initialisation starting"
echo " Target: ${SUPERSET_URL}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 1: Upgrade Superset metadata database schema
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Upgrading Superset metadata database..."
superset db upgrade
echo "  OK"

# ---------------------------------------------------------------------------
# Step 2: Create default roles and permissions
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Initialising default roles..."
superset init
echo "  OK"

# ---------------------------------------------------------------------------
# Step 3: Create admin user (idempotent — skips if already exists)
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Creating admin user '${ADMIN_USER}'..."
superset fab create-admin \
    --username  "${ADMIN_USER}" \
    --firstname "Admin" \
    --lastname  "User" \
    --email     "${ADMIN_MAIL}" \
    --password  "${ADMIN_PASS}" \
    2>&1 | grep -v "already exists" || true
echo "  OK"

# ---------------------------------------------------------------------------
# Step 4: Wait for Superset web process to be ready
#         (This script runs before the superset service container fully starts;
#          we wait until the health endpoint responds.)
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Waiting for Superset web server..."
MAX_ATTEMPTS=60
attempt=0
until wget -qO- "${SUPERSET_URL}/health" > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "${attempt}" -ge "${MAX_ATTEMPTS}" ]; then
        echo "  ERROR: Superset did not become ready after ${MAX_ATTEMPTS} attempts" >&2
        echo "  Skipping dashboard import — run import_dashboards.sh manually when Superset is up." >&2
        exit 0
    fi
    echo "  Waiting (${attempt}/${MAX_ATTEMPTS})..."
    sleep 5
done
echo "  Superset is ready"

# ---------------------------------------------------------------------------
# Step 5: Register datasources via Python script
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Registering datasources..."
python "${DATASOURCES_DIR}/victoriametrics_datasource.py" \
    --superset-url      "${SUPERSET_URL}" \
    --superset-user     "${ADMIN_USER}" \
    --superset-password "${ADMIN_PASS}" \
    --vm-url            "${VM_URL:-http://victoriametrics:8428}" \
    --pg-url            "${APP_DB_URL:-postgresql+psycopg2://app_ro:app_ro@postgres:5432/maintenance}"
echo "  OK"

# ---------------------------------------------------------------------------
# Step 6: Import dashboard JSON files
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Importing dashboards..."

# Log in and capture access token
LOGIN_RESPONSE=$(wget -qO- \
    --header "Content-Type: application/json" \
    --post-data "{\"username\":\"${ADMIN_USER}\",\"password\":\"${ADMIN_PASS}\",\"provider\":\"db\",\"refresh\":true}" \
    "${SUPERSET_URL}/api/v1/security/login" 2>/dev/null)

ACCESS_TOKEN=$(echo "${LOGIN_RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])" 2>/dev/null || echo "")

if [ -z "${ACCESS_TOKEN}" ]; then
    echo "  WARNING: Could not obtain access token — skipping dashboard import."
    echo "  Import manually: POST ${SUPERSET_URL}/api/v1/dashboard/import"
else
    # Fetch CSRF token
    CSRF_TOKEN=$(wget -qO- \
        --header "Authorization: Bearer ${ACCESS_TOKEN}" \
        "${SUPERSET_URL}/api/v1/security/csrf_token/" 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['result'])" 2>/dev/null || echo "")

    import_dashboard() {
        local filename="$1"
        local dashboard_name
        dashboard_name=$(basename "${filename}" .json)

        echo "  Importing: ${dashboard_name}..."

        # Superset dashboard import requires multipart/form-data with a zip file
        # For JSON exports, wrap in a zip first
        local tmpzip="/tmp/${dashboard_name}.zip"
        cd /tmp
        cp "${filename}" "${dashboard_name}.json"
        zip -q "${tmpzip}" "${dashboard_name}.json"
        rm -f "${dashboard_name}.json"

        HTTP_CODE=$(wget -qO- \
            --server-response \
            --header "Authorization: Bearer ${ACCESS_TOKEN}" \
            --header "X-CSRFToken: ${CSRF_TOKEN}" \
            --header "Referer: ${SUPERSET_URL}/" \
            --post-file "${tmpzip}" \
            --header "Content-Type: application/zip" \
            "${SUPERSET_URL}/api/v1/dashboard/import" \
            2>&1 | grep "HTTP/" | tail -1 | awk '{print $2}')

        rm -f "${tmpzip}"

        if [ "${HTTP_CODE}" = "200" ] || [ "${HTTP_CODE}" = "201" ]; then
            echo "    OK (HTTP ${HTTP_CODE})"
        else
            echo "    WARNING: HTTP ${HTTP_CODE} — check Superset logs"
            echo "    You can import manually via:"
            echo "      Dashboards → Import Dashboard → ${filename}"
        fi
    }

    for dashboard_file in "${DASHBOARDS_DIR}"/*.json; do
        if [ -f "${dashboard_file}" ]; then
            import_dashboard "${dashboard_file}"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Superset initialisation complete!"
echo ""
echo " Access Superset at: ${SUPERSET_URL}"
echo " Username: ${ADMIN_USER}"
echo " Password: ${ADMIN_PASS}"
echo ""
echo " Dashboards:"
echo "   Plant OEE Dashboard    → ${SUPERSET_URL}/superset/dashboard/plant-oee/"
echo "   Machine Health         → ${SUPERSET_URL}/superset/dashboard/machine-health/"
echo ""
echo " IMPORTANT: Change the admin password in production!"
echo "   superset fab reset-password --username admin --password <new_password>"
echo "============================================================"
