#!/bin/bash
set -e

# Change to the specified working directory if provided
if [ -n "$WORKDIR" ]; then
  cd "$WORKDIR"
fi

# Execute the command passed to the container
exec "$@"
