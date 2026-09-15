# Source this after configuring native-library paths, if your host needs them.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${ETHOSLM_ENV_FILE:-$ROOT/.env}" ]; then
    source "${ETHOSLM_ENV_FILE:-$ROOT/.env}"
fi
for ethoslm_lib_root in "${NIX_GLIBC:-}" "${NIX_GCCLIB:-}" "${NIX_ZLIB:-}"; do
    if [ -n "$ethoslm_lib_root" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$ethoslm_lib_root/lib:"*) ;;
            *) export LD_LIBRARY_PATH="$ethoslm_lib_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
done
if [ -n "${ETHOSLM_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$ETHOSLM_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset ethoslm_lib_root
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export PY="${ETHOSLM_PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# The JDK this project unpacks under run/, or whatever java is on the PATH. A checkout
# with no run/jdk is the ordinary case for anybody who is not building worlds on this
# host.
if [ -z "${ETHOSLM_JAVA:-}" ]; then
    if [ -x "$ROOT/run/jdk/bin/java" ]; then
        export ETHOSLM_JAVA="$ROOT/run/jdk/bin/java"
    else
        export ETHOSLM_JAVA="$(command -v java || echo java)"
    fi
fi
export ETHOSLM_JAVA_LOADER="${ETHOSLM_JAVA_LOADER:-${NIX_GLIBC:+$NIX_GLIBC/lib/ld-linux-x86-64.so.2}}"

java() {
    if [ -n "$ETHOSLM_JAVA_LOADER" ]; then
        "$ETHOSLM_JAVA_LOADER" "$ETHOSLM_JAVA" "$@"
    else
        "$ETHOSLM_JAVA" "$@"
    fi
}
export -f java
