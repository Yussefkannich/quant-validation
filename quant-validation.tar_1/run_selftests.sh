#!/usr/bin/env bash
# Fuehrt alle Selbsttests aus. Kein Netz, keine API-Schluessel noetig.
set -u
cd "$(dirname "$0")"
fehler=0
for datei in strategien/*.py pruefung/*.py monitor/*.py; do
    printf '\n\033[1m>>> %s\033[0m\n' "$datei"
    if python3 "$datei" --selftest; then
        :
    else
        fehler=$((fehler + 1))
    fi
done
printf '\n'
if [ "$fehler" -eq 0 ]; then
    echo "Alle Selbsttests bestanden."
else
    echo "$fehler Selbsttest(s) fehlgeschlagen."
fi
exit "$fehler"
