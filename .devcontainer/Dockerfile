# Hardened development container based on Red Hat UBI10.
# Security controls are applied as best practice.
# Run `security-audit.sh` inside the container to check applied controls.
FROM registry.access.redhat.com/ubi10/ubi:latest

LABEL org.opencontainers.image.title="My Research Agent Dev Container" \
      org.opencontainers.image.description="Security-hardened Python dev environment on UBI10" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

# Install uv — fast Python package manager (used instead of pip for deps).
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

# Install required development tools — keep the footprint minimal.
RUN dnf install -y \
    python3 python3-pip python3-devel \
    gcc gcc-c++ make git \
    sudo shadow-utils passwd \
    openssl openssl-devel ca-certificates \
    crypto-policies crypto-policies-scripts \
    curl findutils procps-ng util-linux \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Login banner — shown before any interactive session.
COPY .devcontainer/scripts/banner.txt /etc/issue.net
RUN cp /etc/issue.net /etc/issue && chmod 0644 /etc/issue /etc/issue.net

# Harden SSH daemon if it's installed.
# Config is pre-built in scripts/sshd_config: no root login, key-auth only,
# strong ciphers/MACs, MaxAuthTries 3.
COPY .devcontainer/scripts/sshd_config /tmp/sshd_config
RUN if [ -d /etc/ssh ]; then cp /tmp/sshd_config /etc/ssh/sshd_config && chmod 0600 /etc/ssh/sshd_config; fi && \
    rm -f /tmp/sshd_config

# Set system-wide crypto policy to avoid weak algorithms.
RUN if command -v update-crypto-policies &>/dev/null; then \
    update-crypto-policies --set DEFAULT:NO-SHA1 2>/dev/null || \
    update-crypto-policies --set DEFAULT 2>/dev/null || true; \
fi

# Create a non-root developer user. Containers must not run as root.
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid "${USER_GID}" "${USERNAME}" && \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" \
            --create-home --shell /bin/bash \
            --comment "Developer" --password '!' "${USERNAME}" && \
    passwd -l root  # lock root — no direct root login

# Restrict sudo to package management only; log all usage.
RUN printf 'Defaults requiretty\nDefaults use_pty\nDefaults logfile=/var/log/sudo.log\n' \
        > /etc/sudoers.d/00-hardening && \
    chmod 0440 /etc/sudoers.d/00-hardening && \
    printf '%s ALL=(ALL) NOPASSWD: /usr/local/bin/uv, /usr/bin/pip3, /usr/bin/update-ca-trust\n' "${USERNAME}" \
        > "/etc/sudoers.d/${USERNAME}" && \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

# Secure session defaults applied at every login.
#   umask 027        — no world-readable files by default
#   TMOUT=900        — auto-logout after 15 min of inactivity
#   HISTFILE=/dev/null — suppress command history in container
RUN cat > /etc/profile.d/00-hardening.sh << 'PROFILEEOF'
umask 027
TMOUT=900; export TMOUT; readonly TMOUT
HISTSIZE=0; HISTFILESIZE=0; HISTFILE=/dev/null
export HISTSIZE HISTFILESIZE HISTFILE
PATH=/usr/local/bin:/usr/bin:/bin:/home/developer/.local/bin; export PATH
PROFILEEOF

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

# Install project dependencies via uv (reads from pyproject.toml).
COPY pyproject.toml /tmp/pyproject.toml
RUN python3 -c "import tomllib,subprocess; d=tomllib.load(open('/tmp/pyproject.toml','rb')); subprocess.run(['uv','pip','install','--system','--no-cache',*d['project']['dependencies']],check=True)" && \
    rm /tmp/pyproject.toml

# Install the security audit script.
COPY .devcontainer/scripts/security-audit.sh /usr/local/bin/security-audit.sh
RUN chmod 0750 /usr/local/bin/security-audit.sh && \
    chown "root:${USERNAME}" /usr/local/bin/security-audit.sh

WORKDIR /workspace
RUN chown "${USERNAME}:${USERNAME}" /workspace

ENV TMOUT=900 \
    HISTSIZE=0 \
    HISTFILE=/dev/null \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    PATH=/usr/local/bin:/usr/bin:/bin:/home/developer/.local/bin

# Switch to non-root user for all subsequent layers and at runtime.
USER ${USERNAME}

# Build-time safety check — fail the build if still root.
RUN test "$(id -u)" -ne 0 || (echo "ERROR: Container must not run as root" && exit 1)
