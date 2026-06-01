# Production server container for the Research Agent webserver.
# Security-hardened runtime image based on Red Hat UBI10.
# Runs FastAPI + uvicorn as a non-root user on port 8000.
#
# Build:  docker build -t research-agent-server .
# Run:    docker run --rm -p 8000:8000 \
#           -e OPENAI_API_KEY=sk-... \
#           research-agent-server
FROM registry.access.redhat.com/ubi10/ubi:latest

LABEL org.opencontainers.image.title="Research Agent Server" \
      org.opencontainers.image.description="Security-hardened FastAPI + uvicorn server on UBI10" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

# Install uv — fast Python package manager.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Apply security updates.
RUN dnf update -y --security --nobest 2>/dev/null || dnf update -y && \
    dnf clean all && \
    rm -rf /var/cache/dnf /var/log/dnf*

# Remove insecure / unnecessary packages.
RUN dnf remove -y \
    telnet telnet-server rsh rsh-server \
    ypbind ypserv tftp tftp-server \
    talk talk-server vsftpd ftp nfs-utils \
    2>/dev/null || true && \
    dnf clean all

# Install runtime packages only — no compiler, no dev headers, no git.
RUN dnf install -y \
    python3 \
    shadow-utils passwd \
    openssl ca-certificates \
    crypto-policies crypto-policies-scripts \
    curl \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Set system-wide crypto policy to avoid weak algorithms.
RUN if command -v update-crypto-policies &>/dev/null; then \
    update-crypto-policies --set DEFAULT:NO-SHA1 2>/dev/null || \
    update-crypto-policies --set DEFAULT 2>/dev/null || true; \
fi

# Create a non-root app user. Containers must not run as root.
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid "${USER_GID}" "${USERNAME}" && \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" \
            --create-home --shell /bin/bash \
            --comment "App" --password '!' "${USERNAME}" && \
    passwd -l root

# Secure session defaults applied at every login.
#   umask 027        — no world-readable files by default
#   TMOUT=900        — auto-logout after 15 min of inactivity
#   HISTFILE=/dev/null — suppress command history in container
RUN cat > /etc/profile.d/00-hardening.sh << 'PROFILEEOF'
umask 027
TMOUT=900; export TMOUT; readonly TMOUT
HISTSIZE=0; HISTFILESIZE=0; HISTFILE=/dev/null
export HISTSIZE HISTFILESIZE HISTFILE
PATH=/usr/local/bin:/usr/bin:/bin; export PATH
PROFILEEOF

# Install project dependencies via uv (reads from pyproject.toml).
# Dev dependencies (pip-audit, ruff, mypy) are excluded intentionally.
COPY pyproject.toml /tmp/pyproject.toml
RUN python3 -c "import tomllib,subprocess; d=tomllib.load(open('/tmp/pyproject.toml','rb')); subprocess.run(['uv','pip','install','--system','--no-cache',*d['project']['dependencies']],check=True)" && \
    rm /tmp/pyproject.toml

# Strip SUID/SGID from binaries that don't need it.
RUN find /usr/bin /usr/sbin /bin /sbin -type f \( -perm /4000 -o -perm /2000 \) \
    | grep -v -E '^(/usr/bin/sudo|/usr/bin/su|/usr/bin/passwd|/usr/bin/newgrp|/usr/bin/chsh|/usr/bin/chfn|/usr/bin/gpasswd|/usr/bin/ping|/usr/sbin/pam_timestamp_check|/usr/bin/pkexec)$' \
    | xargs -r chmod a-s 2>/dev/null || true

# Remove world-writable bits from /etc and /usr files.
RUN find /etc /usr -type f -perm /002 -not -path '/proc/*' \
    -exec chmod o-w {} \; 2>/dev/null || true

# Enforce secure permissions on critical auth files.
RUN chmod 0644 /etc/passwd /etc/group && \
    { [ -f /etc/shadow ]  && chmod 0000 /etc/shadow  || true; } && \
    { [ -f /etc/gshadow ] && chmod 0000 /etc/gshadow || true; }

# Copy application source. App code is added last to maximise layer cache reuse.
WORKDIR /app
COPY --chown=${USERNAME}:${USERNAME} server.py main.py /app/
COPY --chown=${USERNAME}:${USERNAME} static/ /app/static/

RUN chown "${USERNAME}:${USERNAME}" /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/usr/local/bin:/usr/bin:/bin \
    TMOUT=900 \
    HISTSIZE=0 \
    HISTFILE=/dev/null

USER ${USERNAME}

# Build-time safety check — fail the build if still root.
RUN test "$(id -u)" -ne 0 || (echo "ERROR: Container must not run as root" && exit 1)

EXPOSE 8000

# Health check via the /health endpoint exposed by server.py.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

# API keys are injected at runtime via -e; never bake secrets into the image.
# Override workers with: docker run ... research-agent-server uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
