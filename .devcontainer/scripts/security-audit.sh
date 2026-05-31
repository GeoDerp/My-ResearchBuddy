#!/bin/bash
# Security audit script for the UBI10 dev container.
# Checks common hardening best practices.
# Usage: security-audit.sh [--fix] [--report]
set -euo pipefail

PASS=0; FAIL=0; WARN=0
FIX_MODE=false
REPORT_FILE=""

for arg in "$@"; do
    case $arg in
        --fix)    FIX_MODE=true ;;
        --report) REPORT_FILE="/tmp/security-report-$(date +%Y%m%d-%H%M%S).txt" ;;
    esac
done

log() { echo "$*" | tee -a "${REPORT_FILE:-/dev/null}"; }

pass() { log "[PASS] $1: $2"; ((PASS++)); }
fail() { log "[FAIL] $1: $2"; ((FAIL++)); }
warn() { log "[WARN] $1: $2"; ((WARN++)); }

log "============================================================"
log " Security Audit — $(date)"
log " Host: $(uname -n) | User: $(id)"
log "============================================================"
log ""

log "[ Account & Identity Management ]"

# Non-root execution
if [ "$(id -u)" -ne 0 ]; then
    pass "user:non-root" "Container running as non-root user (uid=$(id -u))"
else
    fail "user:non-root" "Container is running as root — SECURITY VIOLATION"
fi

# Root account locked
if passwd -S root 2>/dev/null | grep -qE ' L | LK '; then
    pass "user:root-locked" "Root account is locked"
else
    warn "user:root-locked" "Cannot verify root lock status (may require elevated access)"
fi

# Secure umask
CURRENT_UMASK=$(umask)
if [ "$CURRENT_UMASK" = "0027" ] || [ "$CURRENT_UMASK" = "027" ] || \
   [ "$CURRENT_UMASK" = "0077" ] || [ "$CURRENT_UMASK" = "077" ]; then
    pass "fs:umask" "Umask is $CURRENT_UMASK (027 or stricter)"
else
    fail "fs:umask" "Umask is $CURRENT_UMASK — should be 027 or stricter"
    if $FIX_MODE; then
        umask 027
        log "  [FIX] Applied umask 027"
    fi
fi

log ""

log "[ File System & Permissions ]"

# No world-writable files in /etc
WW_FILES=$(find /etc -type f -perm /002 2>/dev/null | wc -l)
if [ "$WW_FILES" -eq 0 ]; then
    pass "fs:world-writable" "No world-writable files found in /etc"
else
    fail "fs:world-writable" "$WW_FILES world-writable file(s) found in /etc"
    if $FIX_MODE; then
        find /etc -type f -perm /002 -exec chmod o-w {} \; 2>/dev/null
        log "  [FIX] Removed world-write bits from /etc files"
    fi
fi

# No unowned files in system directories
UNOWNED=$(find /usr /etc /bin /sbin -nouser -o -nogroup 2>/dev/null | wc -l)
if [ "$UNOWNED" -eq 0 ]; then
    pass "fs:unowned-files" "No unowned files in system directories"
else
    warn "fs:unowned-files" "$UNOWNED unowned file(s) found in system directories"
fi

# /etc/passwd permissions 644
PASSWD_PERMS=$(stat -c "%a" /etc/passwd 2>/dev/null || echo "unknown")
if [ "$PASSWD_PERMS" = "644" ]; then
    pass "fs:passwd-perms" "/etc/passwd permissions are 644"
else
    fail "fs:passwd-perms" "/etc/passwd permissions are $PASSWD_PERMS — should be 644"
fi

# /etc/shadow permissions 000/600
SHADOW_PERMS=$(stat -c "%a" /etc/shadow 2>/dev/null || echo "unknown")
if [ "$SHADOW_PERMS" = "000" ] || [ "$SHADOW_PERMS" = "600" ] || \
   [ "$SHADOW_PERMS" = "0" ] || [ "$SHADOW_PERMS" = "400" ]; then
    pass "fs:shadow-perms" "/etc/shadow permissions are $SHADOW_PERMS"
else
    warn "fs:shadow-perms" "/etc/shadow permissions are $SHADOW_PERMS — should be 000/600"
fi

log ""

log "[ Session Security ]"

# Session timeout ≤ 900s
if [ -n "${TMOUT:-}" ] && [ "${TMOUT}" -le 900 ] 2>/dev/null; then
    pass "session:timeout" "TMOUT is set to ${TMOUT}s (≤ 900s)"
else
    fail "session:timeout" "TMOUT is not set or exceeds 900s (current: ${TMOUT:-unset})"
    if $FIX_MODE; then
        export TMOUT=900
        log "  [FIX] Set TMOUT=900"
    fi
fi

# Command history suppressed
if [ "${HISTFILE:-}" = "/dev/null" ] || [ "${HISTSIZE:-1}" -eq 0 ] 2>/dev/null; then
    pass "session:history" "Command history is suppressed"
else
    warn "session:history" "Command history may be enabled (HISTFILE=${HISTFILE:-default})"
fi

log ""

log "[ Network & Services ]"

# SSH daemon hardening, if installed
if command -v sshd &>/dev/null; then
    if grep -q "^PermitRootLogin no" /etc/ssh/sshd_config 2>/dev/null; then
        pass "ssh:no-root-login" "SSH PermitRootLogin is 'no'"
    else
        fail "ssh:no-root-login" "SSH PermitRootLogin is not set to 'no'"
    fi
    if grep -q "^PermitEmptyPasswords no" /etc/ssh/sshd_config 2>/dev/null; then
        pass "ssh:no-empty-passwords" "SSH PermitEmptyPasswords is 'no'"
    else
        fail "ssh:no-empty-passwords" "SSH PermitEmptyPasswords is not set to 'no'"
    fi
    if grep -q "^MaxAuthTries [1-3]$" /etc/ssh/sshd_config 2>/dev/null; then
        pass "ssh:max-auth-tries" "SSH MaxAuthTries is ≤ 3"
    else
        warn "ssh:max-auth-tries" "SSH MaxAuthTries not verified at ≤ 3"
    fi
else
    warn "ssh:hardening" "sshd not installed — SSH checks skipped"
fi

# Prohibited packages must not be installed
for pkg in telnet rsh ypbind tftp talk vsftpd; do
    if rpm -q "$pkg" &>/dev/null; then
        fail "pkgs:prohibited" "Prohibited package installed: $pkg"
    else
        pass "pkgs:prohibited" "Prohibited package not installed: $pkg"
    fi
done

log ""

log "[ Cryptographic Controls ]"

# System crypto policy
if command -v update-crypto-policies &>/dev/null; then
    POLICY=$(update-crypto-policies --show 2>/dev/null || echo "unknown")
    if echo "$POLICY" | grep -qE "FIPS|DEFAULT"; then
        pass "crypto:policy" "Crypto policy is: $POLICY"
    else
        warn "crypto:policy" "Crypto policy is: $POLICY — FIPS or DEFAULT recommended"
    fi
else
    warn "crypto:policy" "update-crypto-policies not available"
fi

# OpenSSL must be present
if command -v openssl &>/dev/null; then
    SSL_VER=$(openssl version 2>/dev/null)
    pass "crypto:openssl" "OpenSSL present: $SSL_VER"
else
    fail "crypto:openssl" "OpenSSL not installed"
fi

log ""

log "[ System Banners ]"

# Login banner must be present
if [ -s /etc/issue.net ]; then
    pass "banner:issue-net" "Login banner (/etc/issue.net) is present"
else
    fail "banner:issue-net" "Login banner (/etc/issue.net) is missing or empty"
fi

if [ -s /etc/issue ]; then
    pass "banner:issue" "Console banner (/etc/issue) is present"
else
    fail "banner:issue" "Console banner (/etc/issue) is missing or empty"
fi

log ""

log "[ Application Security (Python) ]"

# No .pyc bytecode files written
if [ "${PYTHONDONTWRITEBYTECODE:-0}" = "1" ]; then
    pass "app:no-bytecode" "PYTHONDONTWRITEBYTECODE=1 is set"
else
    warn "app:no-bytecode" "PYTHONDONTWRITEBYTECODE not set — .pyc files may be created"
fi

# pip must not run as root
if [ "$(id -u)" -ne 0 ]; then
    pass "app:pip-not-root" "pip will not run as root"
else
    fail "app:pip-not-root" "Running as root — pip installs are a privilege escalation risk"
fi

# Scan for known CVEs in installed packages (requires pip-audit)
if command -v pip-audit &>/dev/null; then
    log "  Running pip-audit vulnerability scan..."
    pip-audit --desc 2>/dev/null && pass "app:pip-audit" "No known CVEs found by pip-audit" \
                                   || warn "app:pip-audit" "pip-audit reported issues — review above"
else
    warn "app:pip-audit" "pip-audit not installed — install with: pip install pip-audit"
fi

log ""
log "============================================================"
log " AUDIT SUMMARY"
log "============================================================"
log " PASS: $PASS  |  FAIL: $FAIL  |  WARN: $WARN"
log "============================================================"

if [ "${REPORT_FILE:-}" != "" ]; then
    log ""
    log "Report saved to: $REPORT_FILE"
fi

# Exit non-zero if any hard failures found
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
