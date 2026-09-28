#!/usr/bin/env bash
# MAINTAINER ONLY: give artifact reviewers restricted, revocable SSH access to
# the Coral board, without sharing any personal key. Reviewers do not run this.
#
#   eval-access.sh create JUMP_ALIAS BOARD_ALIAS   new evaluation key + eval-ssh/{config,known_hosts}
#   eval-access.sh install                         add the key to the jump host and board (uses YOUR ssh access)
#   eval-access.sh check                           prove the reviewer path works and the restrictions hold
#   eval-access.sh seal                            encrypt .env + eval-ssh/ into credentials.tar.gz.enc
#   eval-access.sh revoke                          remove the key from both hosts again
#
# JUMP_ALIAS / BOARD_ALIAS are Host aliases from your own ~/.ssh/config: the
# publicly reachable jump host, and the board behind it. The evaluation key
# can only (a) tunnel through the jump host to the board's SSH port and
# (b) log in to the board as its user, and only when coming from the jump host.
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=eval-ssh
key=$dir/id_ed25519_beaver_eval
tag=beaver-edge-eval                          # comment on every installed key line
image=${BEAVER_EDGE_IMAGE:-beaver-edge-artifact:middleware26}
sealed=credentials.tar.gz.enc
password_file=.credentials-password
iterations=600000                              # must match run-artifact.sh
ssh_options=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o ConnectTimeout=20)

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 1; }
field() { ssh -G "$1" | awk -v k="$2" '$1 == k { print $2; exit }'; }
load_meta() {
    [[ -f $dir/install/meta ]] || die "no $dir/install/meta - run: $0 create JUMP_ALIAS BOARD_ALIAS"
    # shellcheck disable=SC1091
    source "$dir/install/meta"
}

create() {
    local jump=${1:?usage: create JUMP_ALIAS BOARD_ALIAS} board=${2:?usage: create JUMP_ALIAS BOARD_ALIAS}
    [[ ! -e $dir ]] || die "$dir/ already exists - revoke and delete it first to rotate the key"
    local jump_host jump_port jump_user board_host board_user jump_hostkey board_hostkey board_sees
    jump_host=$(field "$jump" hostname); jump_port=$(field "$jump" port); jump_user=$(field "$jump" user)
    board_host=$(field "$board" hostname); board_user=$(field "$board" user)
    echo "jump host : $jump_user@$jump_host:$jump_port   (alias $jump)"
    echo "board     : $board_user@$board_host:22 behind the jump host   (alias $board)"
    # Host keys are read over your existing, already-verified connections.
    jump_hostkey=$(ssh "${ssh_options[@]}" "$jump" 'cat /etc/ssh/ssh_host_ed25519_key.pub' | awk '{ print $1, $2 }')
    board_hostkey=$(ssh "${ssh_options[@]}" "$board" 'cat /etc/ssh/ssh_host_ed25519_key.pub' | awk '{ print $1, $2 }')
    # The address the board sees when the jump host connects to it.
    board_sees=$(ssh "${ssh_options[@]}" "$jump" "ip route get $board_host" | awk '{ for (i = 1; i < NF; i++) if ($i == "src") { print $(i + 1); exit } }')
    [[ -n $jump_hostkey && -n $board_hostkey && -n $board_sees ]] || die "could not read host keys or the jump host's LAN address"
    echo "jump host reaches the board from $board_sees"

    mkdir -p "$dir/install"
    chmod 700 "$dir"
    ssh-keygen -q -t ed25519 -N '' -C "$tag" -f "$key"
    local pub
    pub=$(awk '{ print $1, $2 }' "$key.pub")
    cat > "$dir/config" <<EOF
# BEAVER-EDGE artifact evaluation: evaluation-only key, revoked after the evaluation.
Host beaver-jump
  HostName $jump_host
  Port $jump_port
  User $jump_user

Host coral
  HostName $board_host
  User $board_user
  ProxyJump beaver-jump

Host beaver-jump coral
  IdentityFile ~/.ssh/$(basename "$key")
  IdentitiesOnly yes
  HostKeyAlgorithms ssh-ed25519
  StrictHostKeyChecking yes
  UpdateHostKeys no
  BatchMode yes
  ConnectTimeout 20
  ServerAliveInterval 30
  ServerAliveCountMax 4
EOF
    local jump_known=$jump_host
    [[ $jump_port == 22 ]] || jump_known="[$jump_host]:$jump_port"
    printf '%s %s\n%s %s\n' "$jump_known" "$jump_hostkey" "$board_host" "$board_hostkey" > "$dir/known_hosts"
    # Jump host: no shell, no command, no agent/X11; may only open a tunnel to the board's SSH port.
    # port-forwarding also re-enables -R, so permitlisten pins it to privileged port 1, which a
    # non-root user cannot bind (sshd rejects the whole key for permitlisten="none").
    echo "restrict,port-forwarding,permitlisten=\"127.0.0.1:1\",permitopen=\"$board_host:22\",command=\"/bin/false\" $pub $tag" > "$dir/install/authorized_keys.jump"
    # Board: normal command execution + SCP (Py-TPU needs both), only when coming from the jump host.
    echo "restrict,from=\"$board_sees\" $pub $tag" > "$dir/install/authorized_keys.board"
    printf 'JUMP_ALIAS=%q\nBOARD_ALIAS=%q\n' "$jump" "$board" > "$dir/install/meta"
    echo
    echo "Created $dir/. Lines to install:"
    echo "  on $jump ($jump_user):  $(cat "$dir/install/authorized_keys.jump")"
    echo "  on $board ($board_user): $(cat "$dir/install/authorized_keys.board")"
    echo "Next: $0 install"
}

# install_on ALIAS FILE: append the key line once, after a one-time backup of authorized_keys.
install_on() {
    local alias=$1 line body
    line=$(cat "$2")
    body=$(awk '{ print $2 }' "$key.pub")
    ssh "${ssh_options[@]}" "$alias" "
        umask 077; mkdir -p ~/.ssh; f=~/.ssh/authorized_keys; touch \$f
        [ -e \$f.pre-$tag ] || cp -p \$f \$f.pre-$tag
        if grep -qF '$body' \$f; then echo 'already installed'; else printf '%s\n' '$line' >> \$f && echo 'installed'; fi
    " | sed "s/^/  $alias: /"
}

remove_from() {  # remove_from ALIAS
    local alias=$1 body
    body=$(awk '{ print $2 }' "$key.pub")
    # cat > keeps the file's owner and mode.
    ssh "${ssh_options[@]}" "$alias" "
        f=~/.ssh/authorized_keys
        if grep -qF '$body' \$f; then grep -vF '$body' \$f > \$f.tmp-$tag; cat \$f.tmp-$tag > \$f; rm -f \$f.tmp-$tag; echo 'removed'; else echo 'not present'; fi
    " | sed "s/^/  $alias: /"
}

check() {
    [[ -f $key && -f $dir/config && -f $dir/known_hosts ]] || die "run create first"
    [[ -f .env ]] || die "missing .env (needs REMOTE_HOST=coral and the REMOTE_* board paths)"
    docker image inspect "$image" >/dev/null 2>&1 || die "image $image not found - run ./build-image.sh"
    # Exactly the reviewer path: only eval-ssh/ is mounted, never ~/.ssh.
    # scripts/ is mounted so the check uses this checkout's scripts even if the image is older.
    # EVAL_ACCESS_NETWORK is only for the simulated-hosts self-test.
    docker run --rm ${EVAL_ACCESS_NETWORK:+--network "$EVAL_ACCESS_NETWORK"} -v "$PWD/$dir:/run/evaluation-ssh:ro" -v "$PWD/.env:/artifact/.env:ro" \
        -v "$PWD/scripts:/artifact/scripts:ro" "$image" bash -c '
        echo "== reviewer path (inside the container, evaluation key only)"
        bash scripts/container/setup-ssh-and-run-task.sh --check-board; status=$?
        echo "== restrictions on the jump host"
        options=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes)
        out=$(timeout 30 ssh "${options[@]}" beaver-jump "echo SHELL_OPEN" 2>&1)
        if grep -q SHELL_OPEN <<<"$out"; then echo "FAIL  jump host gave the evaluation key a shell"; status=1
        else echo "ok    jump host refuses a shell/command"; fi
        for target in 127.0.0.1:22 1.1.1.1:443; do
            out=$(timeout 30 ssh "${options[@]}" -W "$target" beaver-jump </dev/null 2>&1 | head -c 300)
            if grep -q "^SSH-" <<<"$out" || ! grep -qi "prohibited\|open failed" <<<"$out"; then
                echo "FAIL  jump host forwarded to $target: $out"; status=1
            else echo "ok    jump host refuses a tunnel to $target"; fi
        done
        out=$(timeout 30 ssh "${options[@]}" -N -o ExitOnForwardFailure=yes -R 127.0.0.1:18022:127.0.0.1:22 beaver-jump 2>&1)
        if grep -q "remote port forwarding failed" <<<"$out"; then echo "ok    jump host refuses a listening port (-R)"
        else echo "FAIL  jump host accepted a listening port (-R): $out"; status=1; fi
        exit $status'
}

seal() {
    [[ -f .env ]] || die "missing .env"
    [[ -f $key ]] || die "run create first"
    docker image inspect "$image" >/dev/null 2>&1 || die "image $image not found - run ./build-image.sh"
    grep -q '^REMOTE_HOST=.*coral' .env || echo "warning: .env does not set REMOTE_HOST=coral (the alias in $dir/config)" >&2
    if [[ -z ${BEAVER_EDGE_CREDENTIALS_PASSWORD:-} ]]; then
        if [[ -s $password_file ]]; then
            BEAVER_EDGE_CREDENTIALS_PASSWORD=$(cat "$password_file")
        else
            BEAVER_EDGE_CREDENTIALS_PASSWORD=$(openssl rand -base64 30 | tr -d '/+=\n' | cut -c1-24)
            (umask 077; printf '%s\n' "$BEAVER_EDGE_CREDENTIALS_PASSWORD" > "$password_file")
        fi
    fi
    export BEAVER_EDGE_CREDENTIALS_PASSWORD
    # openssl runs in the image (the same OpenSSL that run-artifact.sh decrypts with).
    COPYFILE_DISABLE=1 tar -czf - .env "$dir/config" "$dir/known_hosts" "$key" "$key.pub" \
        | docker run --rm -i -e BEAVER_EDGE_CREDENTIALS_PASSWORD "$image" \
            openssl enc -aes-256-cbc -pbkdf2 -iter "$iterations" -salt -pass env:BEAVER_EDGE_CREDENTIALS_PASSWORD \
        > "$sealed.tmp"
    local listing
    listing=$(docker run --rm -i -e BEAVER_EDGE_CREDENTIALS_PASSWORD "$image" \
        openssl enc -d -aes-256-cbc -pbkdf2 -iter "$iterations" -pass env:BEAVER_EDGE_CREDENTIALS_PASSWORD < "$sealed.tmp" | tar -tzf -) \
        || { rm -f "$sealed.tmp"; die "round-trip decryption failed"; }
    mv "$sealed.tmp" "$sealed"
    echo "Sealed into $sealed ($(wc -c < "$sealed" | tr -d ' ') bytes):"
    sed 's/^/  /' <<<"$listing"
    echo "Password: $password_file (give it to reviewers through the submission system, never in the repository)"
}

case ${1:-} in
    create) shift; create "$@" ;;
    install) load_meta; install_on "$JUMP_ALIAS" "$dir/install/authorized_keys.jump"; install_on "$BOARD_ALIAS" "$dir/install/authorized_keys.board" ;;
    revoke) load_meta; remove_from "$JUMP_ALIAS"; remove_from "$BOARD_ALIAS" ;;
    check) check ;;
    seal) seal ;;
    -h|--help|'') usage ;;
    *) usage >&2; exit 2 ;;
esac
