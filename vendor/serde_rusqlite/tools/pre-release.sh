#!/bin/bash

set -e

README_TPL="$WORKSPACE_ROOT/README.tpl"
README="$CRATE_ROOT/README.md"

if [[ "$DRY_RUN" == "false" ]]; then
	cargo readme --template="$README_TPL" --output="$README"
else
	echo "Dry run, would generate $README from $README_TPL"
fi
